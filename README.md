```
#---------------------------相对的改进，如需训练先执行annotation.py划分出本地的图片集合，纯推理预测则不用------------------------#
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4
```
颈网部分
```
#------------------------(backbone.py)-----------------#
class Backbone(nn.Module):
    # self.dark5 = nn.Sequential()
    self.dark6 = Conv(256,256,3,2)
    def forward(self, x):
        # x = self.dark2(x)
        feat0 = x
        # feat3 = x
        feat4 = self.dark6(feat3)
        return feat1, feat2, feat3, feat0, feat4 #多补了后两图

#------------------------(yolo.py)-----------------#
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

class YoloBody(nn.Module):
    # self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.上采=nn.Upsample(scale_factor=2); self.终末卷=Cb(256+256,256)
        self.末中卷=Cb(256+128,128); self.中首卷=Cb(128+64,64)#代后“加强特征提取网络”间内容
        self.初图下采 = Conv(32,32,3,2); self.初首卷 = Cb(32+64,64,0)        
        self.首图下采 = Conv(64,64,3,2); self.首中卷 = Cb(64+128,128,0)
        self.中图下采 = Conv(128,128,3,2); self.中末卷 = Cb(128+256,256,0)
        # self.stride = torch.tensor([256/x.shape[-2] for x in self.backbone.forward(torch.zeros(1,3,256,256))])
        self.stride = self.stride[:3]
    def forward(self, x):
        首图, 中图, 末图, 初图, 终图 = self.backbone.forward(x)
        类末图 = self.终末卷(torch.cat([self.上采(终图),末图],1))
        类中图 = self.末中卷(torch.cat([self.上采(类末图),中图],1))
        类首图 = self.中首卷(torch.cat([self.上采(类中图),首图],1))
        位首图 = self.初首卷(torch.cat([self.初图下采(初图),类首图+首图],1))
        位中图 = self.首中卷(torch.cat([self.首图下采(位首图),类中图+中图],1))
        位末图 = self.中末卷(torch.cat([self.中图下采(位中图),类末图+末图],1))
        #----------------------加强特征提取网络------------#
        shape = 末图.shape #shape及shape后两句连着替换
        x = [位首图,位中图,位末图]; x0 = [类首图,类中图,类末图]
        for i in range(3):x[i]=torch.cat((self.cv2[i](x[i]),self.cv3[i](x0[i])),1)

#------------------------(yolo.py)-----------------#
        "model_path"        : '无注733.pth',#自训的

#------------------------(train.ipynb)-----------------#
        "model_path"        : '无注733.pth',#自训的

#--------------------(gradcam.ipynb)--------------#
        #生成图片（默认就是voc文件夹中的4.jpg）经网络某显著层时的梯度热力图于result.png
```