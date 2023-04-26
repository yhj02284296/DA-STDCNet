#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Module, Sequential, Conv2d, Parameter

from nets.bisestdcnet import STDCNet1446, STDCNet813,BiSeSTDCNet,BiSeSTDC2
#from modules.bn import InPlaceABNSync as BatchNorm2d
BatchNorm2d = nn.BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)#映射到对应类别;是否可分离卷积？
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()

        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)

        self.bn_atten = BatchNorm2d(out_chan)
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')

        # self.sigmoid_atten = nn.Sigmoid()
        self.tanh_atten = nn.Tanh()   #更换激活函数
        self.init_weight()

    def forward(self, x):

        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.tanh_atten(atten)
        out = torch.mul(feat, atten)
        return out + feat  #添加残差连接

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        # self.softmax = Softmax(dim=-1)
        self.tanh_atten=nn.Tanh()
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.tanh_atten(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        # self.softmax  = Softmax(dim=-1)
        self.tanh_atten = nn.Tanh()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.tanh_atten(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

#引入DAM模块
class DAM_Module(nn.Module):
    def __init__(self,in_chan, out_chan,):
        super(DAM_Module, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.attention_ca = CAM_Module(in_dim=out_chan)
        self.attention_sa = PAM_Module(in_dim=out_chan)

    def forward(self, x):
        feat = self.conv(x)
        out_ca = self.attention_ca(feat)
        out_sa = self.attention_sa(feat)
        return out_ca + out_sa + feat


class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.backbone_name = backbone

        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)##stage4,512imput
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)##stage5,1024imput
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        elif backbone == 'BiSeSTDCNet':
            #
            self.backbone = BiSeSTDCNet(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            # self.arm16 = AttentionRefinementModule(512, 128)  ##stage4,512imput
            # self.arm8 = AttentionRefinementModule(256, 128)  ##256=stage3 out channal
            self.arm16 = DAM_Module(512, 128)  #使用DAM替换ARM
            self.arm8 = DAM_Module(256, 128)  #

            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            # self.arm32 = AttentionRefinementModule(inplanes, 128)  ##stage5,1024imput
            self.arm32 = DAM_Module(inplanes, 128)

            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head8 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)#xyy add 4.15
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        elif backbone == 'BiSeSTDC2':
            #
            self.backbone = BiSeSTDC2(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)  ##stage4,512imput
            self.arm8 = AttentionRefinementModule(256, 128)  ##256=stage3 out channal
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)  ##stage5,1024imput
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head8 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)#xyy add 4.15
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        if self.backbone_name == 'BiSeSTDCNet':
            feat2, feat4, feat8, feat16, feat32,feat8sp = self.backbone(x)
        elif self.backbone_name == 'BiSeSTDC2':
            feat2, feat4, feat8, feat16, feat32, feat8sp = self.backbone(x)
        else:
            feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        H8, W8 = feat8.size()[2:]
        H4, W4 = feat4.size()[2:]
        if self.backbone_name == 'BiSeSTDCNet':
            H8SP,W8SP = feat8sp.size()[2:]
        elif  self.backbone_name == 'BiSeSTDC2':
            H8SP, W8SP = feat8sp.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)#conv，bn,relu,output128
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)#
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')#
        feat32_up = self.conv_head32(feat32_up)#conv，bn,relu,output128

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up#stage4=,
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        if self.backbone_name == 'BiSeSTDCNet':
            # feat8_up
            feat8_arm = self.arm8(feat8)
            feat8_sum = feat8_arm + feat16_up
            feat8_up = F.interpolate(feat8_sum, (H4, W4), mode='nearest')
            feat8_up = self.conv_head8(feat8_up)#
            return feat2, feat4, feat8, feat16, feat8_up,feat16_up, feat32_up,feat8sp # bnet,feat2,4,8,18;cp8,cp16
        elif self.backbone_name == 'BiSeSTDC2':
            feat8_arm = self.arm8(feat8)
            feat8_sum = feat8_arm + feat16_up
            feat8_up = F.interpolate(feat8_sum, (H4, W4), mode='nearest')
            feat8_up = self.conv_head8(feat8_up)
            return feat2, feat4, feat8, feat16, feat8_up, feat16_up, feat32_up, feat8sp  # bnet,feat2,4,8,18;cp8,cp16
        else:
            return feat2, feat4, feat8, feat16, feat16_up, feat32_up # bnet,feat2,4,8,18;cp8,cp16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.backbone = backbone
        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8#true
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)
        self.backbone = backbone
        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes##384
        elif backbone == 'BiSeSTDCNet':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes  ##384
        elif backbone == 'BiSeSTDC2':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes  ##384
        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusionModule(inplane, 256)

        self.conv_out = BiSeNetOutput(256, 256, n_classes)#imput ,mid channal,n_class,result_channanl
        self.conv_out8 = BiSeNetOutput(128, 128, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)
        
        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        if self.backbone == 'BiSeSTDCNet':
            feat_res2, feat_res4, feat_res8, feat_res16, feat_cp4, feat_cp8, feat_cp16,feat_sp8 = self.cp(x)
        elif self.backbone == 'BiSeSTDC2':
            feat_res2, feat_res4, feat_res8, feat_res16, feat_cp4, feat_cp8, feat_cp16, feat_sp8 = self.cp(x)
        else:
            feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)
        feat_out_sp4 = self.conv_out_sp4(feat_res4)

        if self.backbone == 'BiSeSTDCNet':
            feat_out_sp8 = self.conv_out_sp8(feat_sp8)#select from spation branch
        elif self.backbone == 'BiSeSTDC2':
            feat_out_sp8 = self.conv_out_sp8(feat_sp8)  # select from spation branch
        else:
            feat_out_sp8 = self.conv_out_sp8(feat_res8)

        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        if self.backbone == 'BiSeSTDCNet':
            feat_fuse = self.ffm(feat_sp8, feat_cp4)
        elif self.backbone == 'BiSeSTDC2':
            feat_fuse = self.ffm(feat_sp8, feat_cp4)
        else:
            feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)

        feat_out8 = self.conv_out8(feat_cp4)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)##first out result
        feat_out8 =  F.interpolate(feat_out8, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)


        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8
        if (not self.use_boundary_2) and self.use_boundary_4 and (not self.use_boundary_8):#
            return feat_out, feat_out16, feat_out32, feat_out_sp4,feat_out8
        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp8,feat_out8#?feat_out_sp8= detail8
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16, feat_out32,feat_out8

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    
    net = BiSeNet('BiSeSTDCNet', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32, sp8 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

    
