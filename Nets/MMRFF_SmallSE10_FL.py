import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

grid_size = 16
eps = 1e-10


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, Act_Setting=0, stride=1, b_bn=True):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size//2, bias=False)
        self.b_bn = b_bn
        if b_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if Act_Setting == 0:
            self.activation = nn.ReLU()
        elif Act_Setting == 1:
            self.activation = nn.ReLU6()
        elif Act_Setting == 2:
            self.activation = Hswish()

    def forward(self, x):
        x = self.conv(x)
        if self.b_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()

        if channel < reduction:
            reduction = 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Act_Setting, FirstSize=3, b_local=True, b_left_BN=False, b_BLock_SE=False):
        super(BasicBlock, self).__init__()

        self.out_channels = out_channels
        self.b_local = b_local
        #
        self.conv0 = BasicConv(in_channels, out_channels, FirstSize, Act_Setting=Act_Setting, b_bn=True)  #
        self.conv1 = BasicConv(out_channels, out_channels, 3, Act_Setting=Act_Setting, b_bn=b_left_BN)  #
        self.conv2 = BasicConv(out_channels, out_channels, 3, Act_Setting=Act_Setting, b_bn=b_left_BN)  #
        self.conv3 = BasicConv(out_channels, out_channels, 1, Act_Setting=Act_Setting, b_bn=b_left_BN)  #

        #
        self.b_BLock_SE = b_BLock_SE
        if b_BLock_SE:
            self.SE = SEModule(out_channels, reduction=2)  # ch attention
        #

    def forward(self, x):
        x = self.conv0(x)

        if self.b_BLock_SE:
            b, c, _, _ = x.size()  #
            se_x = self.SE(x).view(b, c, 1, 1)
        else:
            se_x = None  #

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        if self.b_BLock_SE:
            x3 = x3 * se_x.expand_as(x3)  #

        if self.b_local:
            # res
            r1 = x + x1  #
            r2 = x + x2
            if self.b_BLock_SE:
                r2 = r2 * se_x.expand_as(r2)  #
            r3 = x + x3
            res = torch.cat((r1, r2, r3), dim=1)  #
            return res, x3

        return None, x3


class Yolo_head(nn.Module):
    def __init__(self, obj_ch_sum, reg_ch_sum, obj_out_ch, reg_out_ch, Act_Setting=0,):
        super(Yolo_head, self).__init__()

        self.head_conv1 = BasicConv(obj_ch_sum, obj_out_ch, 3, Act_Setting=Act_Setting)
        self.head_conv2 = BasicConv(reg_ch_sum, reg_out_ch, 3, Act_Setting=Act_Setting)

        self.conv_j2 = nn.Conv2d(obj_out_ch, 30, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_j3 = nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_r2 = nn.Conv2d(reg_out_ch, 45, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_r3 = nn.Conv2d(45, 4, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, obj_fea, reg_fea):
        head_fea1 = self.head_conv1(obj_fea)
        head_fea2 = self.head_conv2(reg_fea)

        obj = self.conv_j2(head_fea1)
        obj = self.conv_j3(obj)

        reg = self.conv_r2(head_fea2)
        reg = self.conv_r3(reg)

        return obj, reg


class YoloBody(nn.Module):
    def __init__(self, Act_Setting=0, ch_setting=(2, 4, 4, 8, 1),
                 b_global=False, b_local=False, b5=False, b_left_BN=False, bMaxAct=False,
                 b_Hsigmoid=False,  obj_head=(0.2, 0.5, 1, 1, 1),
                 reg_head=(1, 1, 1, 1, 1),
                 b_BLock_SE=False, b_cuda=True):
        super(YoloBody, self).__init__()

        self.b_global = b_global  #
        self.b5 = b5  #
        self.b_local = b_local  #
        self.bMaxAct = bMaxAct  #
        self.obj_head = obj_head  #
        self.reg_head = reg_head  #
        self.b_cuda = b_cuda  #
        assert(b_global or b_local)  #

        ch_out1 = ch_setting[0]
        ch_out2 = ch_setting[1]
        ch_out3 = ch_setting[2]
        ch_out4 = ch_setting[3]
        ch_out5 = ch_setting[4]

        self.block1 = BasicBlock(1, ch_out1, Act_Setting, FirstSize=3, b_local=b_local, b_left_BN=b_left_BN, b_BLock_SE=b_BLock_SE)
        self.block2 = BasicBlock(ch_out1, ch_out2, Act_Setting, FirstSize=1, b_local=b_local, b_left_BN=b_left_BN, b_BLock_SE=b_BLock_SE)
        self.block3 = BasicBlock(ch_out2, ch_out3, Act_Setting, FirstSize=1, b_local=b_local, b_left_BN=b_left_BN, b_BLock_SE=b_BLock_SE)
        self.block4 = BasicBlock(ch_out3, ch_out4, Act_Setting, FirstSize=1, b_local=b_local, b_left_BN=b_left_BN, b_BLock_SE=b_BLock_SE)
        if self.b5:
            self.block5 = BasicBlock(ch_out4, ch_out5, Act_Setting, FirstSize=1, b_local=b_local, b_left_BN=b_left_BN, b_BLock_SE=b_BLock_SE)

        #
        self.max_pool = nn.MaxPool2d([2, 2], [2, 2])
        if self.bMaxAct:
            self.f1_act = BasicConv(ch_out1, ch_out1, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.f2_act = BasicConv(ch_out2, ch_out2, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.f3_act = BasicConv(ch_out3, ch_out3, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.f4_act = BasicConv(ch_out4, ch_out4, 1, Act_Setting=Act_Setting, b_bn=False)  #

            # 这一组用于原始图像的降采样激活，即残差前，知否要来一次非线性激活
            self.I1_act = BasicConv(1, 1, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.I2_act = BasicConv(1, 1, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.I3_act = BasicConv(1, 1, 1, Act_Setting=Act_Setting, b_bn=False)  #
            self.I4_act = BasicConv(1, 1, 1, Act_Setting=Act_Setting, b_bn=False)  #
            if self.b5:
                self.f5_act = BasicConv(ch_out5, ch_out5, 1, Act_Setting=Act_Setting,  b_bn=False)  #
                self.I5_act = BasicConv(1, 1, 1, Act_Setting=Act_Setting, b_bn=False)  #

        obj_ch_sum = 0
        reg_ch_sum = 0
        if self.b_local:
            #
            ch_out1_g = ch_out1 * 3
            self.grid_down1 = nn.Sequential(
                nn.Conv2d(ch_out1_g * 1, ch_out1_g * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1_g * 2, ch_out1_g * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1_g * 2, ch_out1_g * 4, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1_g * 4, ch_out1_g * 4, 2, 2, 0, bias=False),
            )
            ch_out2_g = ch_out2 * 3
            self.grid_down2 = nn.Sequential(
                nn.Conv2d(ch_out2_g * 1, ch_out2_g * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out2_g * 2, ch_out2_g * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out2_g * 2, ch_out2_g * 4, 2, 2, 0, bias=False),
            )
            ch_out3_g = ch_out3 * 3
            self.grid_down3 = nn.Sequential(
                nn.Conv2d(ch_out3_g * 1, ch_out3_g * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out3_g * 2, ch_out3_g * 4, 2, 2, 0, bias=False),
            )
            ch_out4_g = ch_out4 * 3
            self.grid_down4 = nn.Sequential(
                nn.Conv2d(ch_out4_g * 1, ch_out4_g * 2, 2, 2, 0, bias=False),
            )
            ch_out5_g = ch_out5 * 3

            if not self.b5:
                ch_out5_g = 0
            #
            self.L_head = np.array((ch_out1_g * 4, ch_out2_g * 4, ch_out3_g * 4, ch_out4_g * 2, ch_out5_g))
            #
            self.Obj_L_head = np.round(self.L_head * self.obj_head)
            self.Obj_L_head = self.Obj_L_head.astype(int)
            obj_Lch_sum = sum(self.Obj_L_head[:])
            #
            self.Reg_L_head = np.round(self.L_head * self.reg_head)
            self.Reg_L_head = self.Reg_L_head.astype(int)
            reg_Lch_sum = sum(self.Reg_L_head[:])

            obj_ch_sum = obj_ch_sum + obj_Lch_sum  #
            reg_ch_sum = reg_ch_sum + reg_Lch_sum  #

        if self.b_global:

            if not self.b5:
                ch_out5 = 0
            #
            self.G_head = np.array((ch_out1 * 4, ch_out2 * 4, ch_out3 * 4, ch_out4 * 2, ch_out5))
            #
            self.Obj_G_head = np.round(self.G_head * self.obj_head)
            self.Obj_G_head = self.Obj_G_head.astype(int)
            obj_Gch_sum = sum(self.Obj_G_head[:])
            #
            self.Reg_G_head = np.round(self.G_head * self.reg_head)
            self.Reg_G_head = self.Reg_G_head.astype(int)
            reg_Gch_sum = sum(self.Reg_G_head[:])

            obj_ch_sum = obj_ch_sum + obj_Gch_sum  #
            reg_ch_sum = reg_ch_sum + reg_Gch_sum  #

            self.o_down1 = nn.Sequential(
                nn.Conv2d(ch_out1 * 1, ch_out1 * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1 * 2, ch_out1 * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1 * 2, ch_out1 * 4, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out1 * 4, ch_out1 * 4, 2, 2, 0, bias=False),
            )

            self.o_down2 = nn.Sequential(
                nn.Conv2d(ch_out2 * 1, ch_out2 * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out2 * 2, ch_out2 * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out2 * 2, ch_out2 * 4, 2, 2, 0, bias=False),
            )

            self.o_down3 = nn.Sequential(
                nn.Conv2d(ch_out3 * 1, ch_out3 * 2, 2, 2, 0, bias=False),
                nn.Conv2d(ch_out3 * 2, ch_out3 * 4, 2, 2, 0, bias=False),
            )

            self.o_down4 = nn.Sequential(
                nn.Conv2d(ch_out4 * 1, ch_out4 * 2, 2, 2, 0, bias=False),
            )

        #
        #
        self.Yolo_head = Yolo_head(obj_ch_sum, reg_ch_sum, 100, 100, Act_Setting=Act_Setting)

        if b_Hsigmoid:
            # 0-1
            self.sigmoid = Hsigmoid()  #
        else:
            self.sigmoid = torch.nn.Sigmoid()   #

        self._initialize_weights()

    def forward(self, x, bGet_fea=False):
        #
        x = x.unsqueeze(dim=1)

        # backbone
        res1, fea1 = self.block1(x)

        fea2 = self.max_pool(fea1)
        res2, fea2 = self.block2(fea2)

        fea3 = self.max_pool(fea2)
        res3, fea3 = self.block3(fea3)

        fea4 = self.max_pool(fea3)
        res4, fea4 = self.block4(fea4)

        #
        if self.b5:
            fea5 = self.max_pool(fea4)
            res5, fea5 = self.block5(fea5)
        else:
            res5 = None
            fea5 = None

        #
        obj_local_fea = None
        reg_local_fea = None
        obj_global_fea = None
        reg_global_fea = None

        if self.b_global:
            #
            if self.b5:
                or_1, or_2, or_3, or_4, or_5 = self.get_OriMap(x)
            else:
                or_1, or_2, or_3, or_4 = self.get_OriMap(x)
                or_5 = None

            if self.bMaxAct:
                #
                fea1 = self.f1_act(fea1)
                fea2 = self.f2_act(fea2)
                fea3 = self.f3_act(fea3)
                fea4 = self.f4_act(fea4)
                if self.b5:
                    fea5 = self.f4_act(fea5)
            #
            or_1 = or_1 + fea1
            or_2 = or_2 + fea2
            or_3 = or_3 + fea3
            or_4 = or_4 + fea4
            if self.b5:
                or_5 = or_5 + fea5

            # grid resample operation
            od_1 = self.o_down1(or_1)
            od_2 = self.o_down2(or_2)
            od_3 = self.o_down3(or_3)
            od_4 = self.o_down4(or_4)

            if self.b5:
                obj_global_fea = torch.cat((od_1[:, 0:self.Obj_G_head[0], :, :],
                                            od_2[:, 0:self.Obj_G_head[1], :, :],
                                            od_3[:, 0:self.Obj_G_head[2], :, :],
                                            od_4[:, 0:self.Obj_G_head[3], :, :],
                                            or_5[:, 0:self.Obj_G_head[4], :, :]), dim=1)
                #
                reg_global_fea = torch.cat((od_1[:, self.G_head[0]-self.Reg_G_head[0]:self.G_head[0], :, :],
                                            od_2[:, self.G_head[1]-self.Reg_G_head[1]:self.G_head[1], :, :],
                                            od_3[:, self.G_head[2]-self.Reg_G_head[2]:self.G_head[2], :, :],
                                            od_4[:, self.G_head[3]-self.Reg_G_head[3]:self.G_head[3], :, :],
                                            or_5[:, self.G_head[4]-self.Reg_G_head[4]:self.G_head[4], :, :]), dim=1)
            else:
                obj_global_fea = torch.cat((od_1[:, 0:self.Obj_G_head[0], :, :],
                                            od_2[:, 0:self.Obj_G_head[1], :, :],
                                            od_3[:, 0:self.Obj_G_head[2], :, :],
                                            od_4[:, 0:self.Obj_G_head[3], :, :]), dim=1)
                reg_global_fea = torch.cat((od_1[:, self.G_head[0]-self.Reg_G_head[0]:self.G_head[0], :, :],
                                            od_2[:, self.G_head[1]-self.Reg_G_head[1]:self.G_head[1], :, :],
                                            od_3[:, self.G_head[2]-self.Reg_G_head[2]:self.G_head[2], :, :],
                                            od_4[:, self.G_head[3]-self.Reg_G_head[3]:self.G_head[3], :, :]), dim=1)

        if self.b_local:
            #
            dn1 = self.grid_down1(res1)
            dn2 = self.grid_down2(res2)
            dn3 = self.grid_down3(res3)
            dn4 = self.grid_down4(res4)
            if self.b5:
                dn5 = res5
                obj_local_fea = torch.cat((dn1[:, 0:self.Obj_L_head[0], :, :],
                                           dn2[:, 0:self.Obj_L_head[1], :, :],
                                           dn3[:, 0:self.Obj_L_head[2], :, :],
                                           dn4[:, 0:self.Obj_L_head[3], :, :],
                                           dn5[:, 0:self.Obj_L_head[4], :, :]), dim=1)
                reg_local_fea = torch.cat((dn1[:, self.L_head[0]-self.Reg_L_head[0]:self.L_head[0], :, :],
                                           dn2[:, self.L_head[1]-self.Reg_L_head[1]:self.L_head[1], :, :],
                                           dn3[:, self.L_head[2]-self.Reg_L_head[2]:self.L_head[2], :, :],
                                           dn4[:, self.L_head[3]-self.Reg_L_head[3]:self.L_head[3], :, :],
                                           dn5[:, self.L_head[4]-self.Reg_L_head[4]:self.L_head[4], :, :]), dim=1)
            else:
                obj_local_fea = torch.cat((dn1[:, 0:self.Obj_L_head[0], :, :],
                                           dn2[:, 0:self.Obj_L_head[1], :, :],
                                           dn3[:, 0:self.Obj_L_head[2], :, :],
                                           dn4[:, 0:self.Obj_L_head[3], :, :]), dim=1)
                reg_local_fea = torch.cat((dn1[:, self.L_head[0]-self.Reg_L_head[0]:self.L_head[0], :, :],
                                           dn2[:, self.L_head[1]-self.Reg_L_head[1]:self.L_head[1], :, :],
                                           dn3[:, self.L_head[2]-self.Reg_L_head[2]:self.L_head[2], :, :],
                                           dn4[:, self.L_head[3]-self.Reg_L_head[3]:self.L_head[3], :, :]), dim=1)

        #
        if self.b_global and self.b_local:
            obj_features = torch.cat((obj_global_fea, obj_local_fea), dim=1)
            reg_features = torch.cat((reg_global_fea, reg_local_fea), dim=1)
        else:
            if self.b_local:
                obj_features = obj_local_fea
                reg_features = reg_local_fea
            if self.b_global:
                obj_features = obj_global_fea
                reg_features = reg_global_fea

        obj, reg = self.Yolo_head(obj_features, reg_features)  #

        obj = self.sigmoid(obj)
        reg = self.sigmoid(reg)

        # 这个是为了绘制中间结果而存在的
        if bGet_fea:
            fea_list = list()
            glo_list = list()
            res_list = list()
            fea_list.append(fea1)
            fea_list.append(fea2)
            fea_list.append(fea3)
            fea_list.append(fea4)
            if self.b5:
                fea_list.append(fea5)
            if self.b_local:
                #
                res_list.append(res1)
                res_list.append(res2)
                res_list.append(res3)
                res_list.append(res4)
                if self.b5:
                    res_list.append(res5)
            if self.b_global:
                #
                glo_list.append(or_1)
                glo_list.append(or_2)
                glo_list.append(or_3)
                glo_list.append(or_4)
                if self.b5:
                    glo_list.append(or_5)

            return obj, reg, fea_list, res_list, glo_list, obj_features
        return obj, reg, obj_features  #

    def get_OriMap(self, x):
        #
        if self.bMaxAct:
            #
            or_1 = self.I1_act(x)
            or_2 = self.max_pool(or_1)
            or_2 = self.I2_act(or_2)
            or_3 = self.max_pool(or_2)
            or_3 = self.I3_act(or_3)
            or_4 = self.max_pool(or_3)
            or_4 = self.I4_act(or_4)
            if self.b5:
                or_5 = self.max_pool(or_4)
                or_5 = self.I4_act(or_5)
                return or_1, or_2, or_3, or_4, or_5
            return or_1, or_2, or_3, or_4
        else:
            #
            or_1 = x
            or_2 = self.max_pool(or_1)
            or_3 = self.max_pool(or_2)
            or_4 = self.max_pool(or_3)
            if self.b5:
                or_5 = self.max_pool(or_4)
                return or_1, or_2, or_3, or_4, or_5
            return or_1, or_2, or_3, or_4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()







