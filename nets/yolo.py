import numpy as np
import torch
import torch.nn as nn

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors

# class ChannelAttention(nn.Module):#通道注意力
#     def __init__(self, c):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.f1 = nn.Conv2d(c, c // 16, 1, bias=False)
#         self.f2 = nn.Conv2d(c // 16, c, 1, bias=False)#一核卷相当于进行通道的全连接
#         self.relu = nn.ReLU(); self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
#         max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
#         return self.sigmoid(avg_out + max_out)#得到(b,c,1,1)的通注
# class SpatialAttention(nn.Module):#空间注意力
#     def __init__(self):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         return self.sigmoid(self.conv(x))#得到(b,1,h,w)的空注
# class Cb(nn.Module):      #同时插入这三类
#     def __init__(self, c1, c2, m=1):
#         super(Cb, self).__init__()
#         self.m=m; self.conv=Conv(c1,c2,1,1)
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention()
#     def forward(self, x):#fpn先向下融合时m=1打开通注,后向上融合时为0空注,默认1只需标0
#         if self.m==1:out = self.channel_attention(x) * x
#         if self.m==0:out = self.spatial_attention(x) * x
#         return self.conv(out)#自注意完后就是简简单单的一核卷调整通道

class Cb(nn.Module):      #单独无注意力机制的
    def __init__(self, c1, c2, m=1):
        super(Cb, self).__init__()
        self.conv=Conv(c1,c2,1,1)
    def forward(self, x):
        return self.conv(x)#这里out改为一般的x

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.上采=nn.Upsample(scale_factor=2); self.终末卷=Cb(256+256,256)
        self.末中卷=Cb(256+128,128); self.中首卷=Cb(128+64,64)#代后“加强特征提取网络”间内容
        self.初图下采 = Conv(32,32,3,2); self.初首卷 = Cb(32+64,64,0)        
        self.首图下采 = Conv(64,64,3,2); self.首中卷 = Cb(64+128,128,0)
        self.中图下采 = Conv(128,128,3,2); self.中末卷 = Cb(128+256,256,0)
        #------------------------加强特征提取网络------------------------# 
        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes
        
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        首图, 中图, 末图, 初图, 终图 = self.backbone.forward(x)
        类末图 = self.终末卷(torch.cat([self.上采(终图),末图],1))
        类中图 = self.末中卷(torch.cat([self.上采(类末图),中图],1))
        类首图 = self.中首卷(torch.cat([self.上采(类中图),首图],1))
        位首图 = self.初首卷(torch.cat([self.初图下采(初图),类首图+首图],1))
        位中图 = self.首中卷(torch.cat([self.首图下采(位首图),类中图+中图],1))
        位末图 = self.中末卷(torch.cat([self.中图下采(位中图),类末图+末图],1))
        #------------------------加强特征提取网络------------------------# 
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = 末图.shape #shape及shape后两句连着替换
        x = [位首图,位中图,位末图]; x0 = [类首图,类中图,类末图]
        for i in range(3):x[i]=torch.cat((self.cv2[i](x[i]),self.cv3[i](x0[i])),1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400; 
        #                                           box self.reg_max * 4, 8400
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)