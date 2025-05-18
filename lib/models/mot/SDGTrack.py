import argparse
import math
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.mot.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, ASPP,_NonLocalBlockND
from models.mot.experimental import MixConv2d, CrossConv, C3
from core.mot.general import check_anchor_order, make_divisible, check_file, set_logging
from core.mot.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)

logger = logging.getLogger(__name__)

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), id_embedding=256, ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.id_embedding = id_embedding
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList([nn.Conv2d(ch[1]//2, (self.no) * self.na, 1),
                               nn.Conv2d(ch[2], (self.no) * self.na, 1),
                               nn.Conv2d(ch[0], (self.no) * self.na, 1)])  # output conv
        self.export = False  # onnx export
        self.k = Parameter(torch.ones(1) * 10)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i][0])  # conv
            #x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:  # inference  如果是检测
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()  #对结果进行sigmoid操作
                if self.k[0] == 10:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                else:
                    y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = y[..., :6]
                z.append(y.view(bs, -1, self.no))

        #return [x[0][...,6:],x] if self.training else [x[0][...,6:],(torch.cat(z, 1), x)]
        return [x,self.k] if self.training else (torch.cat(z, 1), [x,self.k]) #返回推理或者训练的结果
#生成网格，用于将预测的相对坐标映射回图像坐标
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class DenseMask(nn.Module):
    def __init__(self,mask=1,ch=()):
        super(DenseMask, self).__init__()
        self.proj1 = Conv(ch[0]//2, 1,k=3)
        self.proj2 = nn.ConvTranspose2d(ch[1], 1, 4, stride=2,
                                                     padding=1, output_padding=0,
                                                     groups=1, bias=False)
        self.proj3 = nn.ConvTranspose2d(ch[2], 1, 8, stride=4,
                                                     padding=2, output_padding=0,
                                                     groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        return self.sigmoid(self.proj1(layers[0][0])+self.proj2(layers[1][0])+self.proj3(layers[2][0]))

class SAAN(nn.Module):
    def __init__(self,id_embedding=256,ch=()):
        super(SAAN, self).__init__()
        self.proj1 = nn.Sequential(Conv(ch[1]//2, 256,k=3),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj2 = nn.Sequential( Conv(ch[2], 256,k=3),
                                    nn.ConvTranspose2d(256, 256, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=256, bias=False),
                                    SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj3 = nn.Sequential(Conv(ch[0], 256,k=3),
                                   nn.ConvTranspose2d(256, 256, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=256, bias=False),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))

        self.node = nn.Sequential(SAAN_Attention(k_size=3, ch=256*3, s_state=False, c_state=True),
                                  Conv(256 * 3, 256,k=3),
                                  nn.Conv2d(256, id_embedding,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=True)
                                  )

    def forward(self, layers):
        #完全体
        layers[0] = self.proj1(layers[0][1])
        layers[1] = self.proj2(layers[1][1])
        layers[2] = self.proj3(layers[2][1])
        #layers[0] = self.proj1(layers[0])
        #layers[1] = self.proj2(layers[1])
        #layers[2] = self.proj3(layers[2])
        id_layer_out = self.node(torch.cat([layers[0], layers[1], layers[2]], 1))
        id_layer_out = id_layer_out.permute(0, 2, 3, 1).contiguous()
        return id_layer_out


class SAAN_Attention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self,k_size = 3,ch = 256, s_state = False, c_state = False):
        super(SAAN_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        #self.conv1 = Conv(ch, ch,k=1)

        self.s_state = s_state
        self.c_state = c_state

        if c_state:
            self.c_attention = nn.Sequential(nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                                                  nn.LayerNorm([1, ch]),
                                                  nn.LeakyReLU(0.3, inplace=True),
                                                  nn.Linear(ch, ch, bias=False))

        if s_state:
            self.conv_s = nn.Sequential(Conv(ch, ch // 4, k=1))
            self.s_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)


    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)+\
                  self.c_attention(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        #spatial_attention
        if self.s_state:
            x_s = self.conv_s(x)
            avg_out = torch.mean(x_s, dim=1, keepdim=True)
            max_out, _ = torch.max(x_s, dim=1, keepdim=True)
            y_s = torch.cat([avg_out, max_out], dim=1)
            y_s = self.sigmoid(self.s_attention(y_s))

        if self.c_state and self.s_state:
            y = x * y_s * y_c + x
        elif self.c_state:
            y = x * y_c + x
        elif self.s_state:
            y = x * y_s + x
        else:
            y = x
        return y

class SELayer(nn.Module):
    def __init__(self, channel):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
                nn.Linear(channel, channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y




class DA(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(DA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(c2, 3)
        self.SE_Layers = nn.ModuleList([SELayer(c2) for _ in range(3)])
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        b, c, _, _ = x.size()

        weight = self.fc_1(self.avg_pool(x).view(b, c))
        weight = self.softmax(weight).view(b, 3, 1)

        for i, SE_Layer in enumerate(self.SE_Layers):
            if i == 0:
                SELayers_Matrix = SE_Layer(x).view(b, c, 1)
            else:
                SELayers_Matrix = torch.cat((SELayers_Matrix, SE_Layer(x).view(b, c, 1)), 2)

        SELayers_Matrix = torch.matmul(SELayers_Matrix, weight).view(b, c, 1, 1)
        SELayers_Matrix = self.sig(SELayers_Matrix)

        return x * SELayers_Matrix



# class IFDR(nn.Module):
#     def __init__(self,k_size = 3,ch=()):
#         super(IFDR, self).__init__()
#         self.conv1_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
#         # self.conv1_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
#         # self.conv2_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         # self.conv2_t = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         self.c_attention_1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))
#         self.c_attention_2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                           nn.InstanceNorm2d(num_features=ch),
#                                           nn.LeakyReLU(0.3, inplace=True))

class IFDR(nn.Module):
    def __init__(self, k_size=3, ch=()):
        super(IFDR, self).__init__()

        self.conv1_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
        self.conv1_2 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
        self.conv1_3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
        self.conv2_3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        # w = 6
        # h = 10
        # # self.max_pool = nn.AdaptiveMaxPool2d((w,h))
        # self.avg_pool = nn.AdaptiveAvgPool2d((w,h))
        # self.c_attention_1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
        #                                    nn.InstanceNorm2d(num_features=ch),
        #                                    nn.LeakyReLU(0.3, inplace=True))
        # self.c_attention_2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
        #                                    nn.InstanceNorm2d(num_features=ch),
        #                                    nn.LeakyReLU(0.3, inplace=True))
# class IFDR(nn.Module):
#     def __init__(self, k_size=3, ch=()):
#         super(IFDR, self).__init__()
#         self.conv1_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
#         # self.conv1_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(ch, ch, kernel_size=1, padding=0)
#         # self.conv2_2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         # self.conv2_t = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
#         self.c_attention_1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                            nn.InstanceNorm2d(num_features=ch),
#                                            nn.BatchNorm2d(ch),  # 加入BatchNorm层
#                                            nn.LeakyReLU(0.3, inplace=True))
#         self.c_attention_2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
#                                            nn.InstanceNorm2d(num_features=ch),
#                                            nn.BatchNorm2d(ch),  # 加入BatchNorm层
#                                            nn.LeakyReLU(0.3, inplace=True))
        # self.w ,self.h = None , None
        # self.avg_pool = nn.AdaptiveAvgPool2d((1 , 1))

    def forward(self, x):
        # 克隆原始输入张量
        # x1 = x.clone()
        # x2 = x.clone()
        b, c, h, w = x.size()
        # self.h = h
        # self.w = w
        # self.avg_pool = nn.AdaptiveAvgPool2d((h, w))
        # x = self.avg_pool(x)
        # 第一部分的操作
        # x= self.avg_pool(x)#加入
        x1 = self.conv1_1(x)
        x1_1 = self.conv1_2(x1)
        x1_1 = self.conv1_3(x1_1)
        # x1_1 = self.conv1_2(x1_1)
        b,c,h,w = x1_1.shape
        # x1_1 = x1_1.contiguous().view(b, c, h * w)
        # x1 = x1.contiguous().view(b, c, h * w)
        # x1_1_T = x1_1.permute(0, 2, 1)
        # x1_1_X = torch.matmul(x1_1,x1_1_T)
        # x1_1_X_softmax = F.softmax(x1_1_X, dim=-1)
        # x1_1_X = torch.matmul(x1_1_X_softmax, x1).contiguous().view(b, c, h, w)
        # 第二部分的操作
        x2 = self.conv2_1(x)
        x2_1 = self.conv2_2(x2)
        x2_1 = self.conv2_3(x2_1)
        # x2_1 = self.conv2_2(x2_1)
        # x2_1 = x2.contiguous().view(b, c, h * w)
        # x2 = x2.contiguous().view(b, c, h * w)
        # x2_1_T = x2_1.permute(0, 2 ,1)
        # x2_1_X = torch.matmul(x2_1,x2_1_T)
        # x2_1_X_softmax = F.softmax(x2_1_X, dim=-1)
        # x2_1_X = torch.matmul(x2_1_X_softmax, x2).contiguous().view(b, c, h, w)
        return [x1_1,x2_1]





class CCN(nn.Module):
    def __init__(self,k_size = 3,ch=()):
        super(CCN, self).__init__()
        #self.independence = 0.7
        #self.share = 0.3
        self.w1 = Parameter(torch.ones(1)*0.5)
        self.w2 = Parameter(torch.ones(1)*0.5)
        w = 6
        h = 10
        # self.max_pool = nn.AdaptiveMaxPool2d((w,h))
        self.avg_pool = nn.AdaptiveAvgPool2d((w,h))

        self.c_attention1 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))
        self.c_attention2 = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.InstanceNorm2d(num_features=ch),
                                          nn.LeakyReLU(0.3, inplace=True))


        self.sigmoid = nn.Sigmoid()
        #self.conv1 = Conv(ch, ch, k=1)
        #self.conv2 = Conv(ch, ch, k=1)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        #改了
        # x=x[1]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) #尝试改改改
        # y_t2_conv = self.c_attention2[1](y)
        # print(y_t2_conv.float())
        y_t1 = self.c_attention1(y)
        y_t2 = self.c_attention2(y)
        bs,c,h,w = y_t1.shape
        y_t1 =y_t1.view(bs, c, h*w)
        y_t2 =y_t2.view(bs, c, h*w)

        y_t1_T = y_t1.permute(0, 2, 1)
        y_t2_T = y_t2.permute(0, 2, 1)
        M_t1 = torch.matmul(y_t1, y_t1_T)
        M_t2 = torch.matmul(y_t2, y_t2_T)
        M_t1 = F.softmax(M_t1, dim=-1)
        M_t2 = F.softmax(M_t2, dim=-1)

        M_s1 = torch.matmul(y_t1, y_t2_T)
        M_s2 = torch.matmul(y_t2, y_t1_T)
        M_s1 = F.softmax(M_s1, dim=-1)
        M_s2 = F.softmax(M_s2, dim=-1)

        x_t1 = x
        x_t2 = x
        bs,c,h,w = x_t1.shape
        x_t1 = x_t1.contiguous().view(bs, c, h*w)
        x_t2 = x_t2.contiguous().view(bs, c, h*w)

        #x_t1 = torch.matmul(self.independence*M_t1 + self.share*M_s1, x_t1).contiguous().view(bs, c, h, w)
        #x_t2 = torch.matmul(self.independence*M_t2 + self.share*M_s2, x_t2).contiguous().view(bs, c, h, w)
        x_t1 = torch.matmul(self.w1*M_t1 + (1-self.w1)*M_s1, x_t1).contiguous().view(bs, c, h, w)
        x_t2 = torch.matmul(self.w2*M_t2 + (1-self.w2)*M_s2, x_t2).contiguous().view(bs, c, h, w)
        #print("M_t1",torch.sort(M_t1[0][0]))
        #print("y_t1",torch.max(y_t1),torch.min(y_t1))
        #print("y_t2", torch.max(y_t2), torch.min(y_t2))
        return [x_t1+x,x_t2+x]



class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict,从原始配置中加载参数赋予self.yaml

        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save, self.out = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out   解析模型parse_model
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect() 模型加载
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            x = self.forward(torch.zeros(2, ch, s, s)) #先用空白数据进行传播
            m.stride = torch.tensor([s / x.shape[-2] for x in x[-1][0]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        output = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers 来自其他层
            #计算计算量（flops）和时间测量并输出
            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))
            #向前传播，保存需要保存的张量
            x = m(x)  # run
            #自己加的
            # if m.__class__.__name__ == 'IFDR':  # 根据模块的类名来决定是否保存输出
            #     x1_1_X,x2_1_X=x[0],x[1]
            #     x=x2_1_X
            if m.i in self.out:
                output.append(x)
            y.append(x if m.i in self.save else None)  # save output
        if profile:
            print('%.1fms total' % sum(dt))
        return output

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            bb = b[:, 4]
            bb = bb + math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            bbb = b[:, 5:]
            bbb = bbb + math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    #上方原来是b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
#b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw,id_embedding = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d['id_embedding']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors如果是列表就是宽高两个参数所以除以二，整数表明只有一组锚点
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # 模型的层列表layers, 保存列表savelist, 获取最新的通道数 ch out
    out_list = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain 通过深度值调整模块的次数
        #如果模块是一些特定的卷积类型，进入条件块
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0] #输入、输出 （通道）

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2 #根据宽度增益 gw 调整输出通道数 c2，确保是8的倍数（硬件效率高）

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2
#下面24改为了30#
            if i == 30:
                args = [c2, c1, *args[1:]]
            else:
                args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            out_list += [i]
        elif m is SAAN:
            out_list += [i]
            args.append([ch[x + 1] for x in f])
        # elif m is DA:
        #     out_list += [i]
        #     c2 = ch[f]
        # #更改了、、、、、、
        # elif m is IFDR:
        #     out_list += [i]
        #     c2 = ch[f]
        elif m is DenseMask:
            out_list += [i]
            args.append([ch[x + 1] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist  用于标记哪些层的输出需要在后续处理中保存
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), sorted(out_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
