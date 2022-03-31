#2022.03.06,by duanyaoming
#Light Field super-resolution using Multi-information attention interaction

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes = args.angRes_in
        channels = 64
        self.factor = args.scale_factor
        self.IntraFeaExtract = FeaExtract(channels)
        self.InterFeaExtract = Extract_inter_fea(channels, self.angRes)

        self.CoordAtt1 = CoordAtt(channels)
        self.VAtt1=VAtt(self.angRes)
        self.CoordAtt2 = CoordAtt(channels)
        self.VAtt2 = VAtt(self.angRes)
        self.CoordAtt3 = CoordAtt(channels)
        self.VAtt3 = VAtt(self.angRes)
        self.CoordAtt4 = CoordAtt(channels)
        self.VAtt4 = VAtt(self.angRes)

        self.conv_fusing_s_c=nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)
        self.conv_fusing_v = fusing_v(channels)
        self.FM=FM(channels,self.angRes)
        #self.FBM=FBM(channels)
        self.Reconstruct = RFDN(channels)
        self.UpSample = Upsample(channels, self.factor)


    def forward(self, x,Lr_info=None):
        #x:(b,1,H,W)[H=angle*h,W=angle*w];x_upscale:(b,1,scale*H,scale*W)
        x_upscale = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False)
        #x_multi:(b,n,h, w)[n=angle*angle]
        x_multi = LFsplit(x, self.angRes)
        #intra_fea_initial:(b, n, c, h, w)
        #inter_fea_initial:(b, c, h, w)
        intra_fea_initial = self.IntraFeaExtract(x_multi)
        inter_fea_initial = self.InterFeaExtract(x_multi)
        #fea_s_c1:(b, c, h, w)
        #fea_v1:(b, n, c, h, w)
        fea_s_c1=self.CoordAtt1(inter_fea_initial)
        fea_v1=self.VAtt1(intra_fea_initial)
        fea_s_c2 = self.CoordAtt2(fea_s_c1)
        fea_v2 = self.VAtt2(fea_v1)
        fea_s_c3 = self.CoordAtt3(fea_s_c2)
        fea_v3 = self.VAtt3(fea_v2)
        fea_s_c4 = self.CoordAtt4(fea_s_c3)
        fea_v4 = self.VAtt4(fea_v3)
        # fea_s_c:(b, 4c, h, w)
        # fea_v:(b, n, 4c, h, w)
        fea_s_c_cat=torch.cat((fea_s_c1,fea_s_c2,fea_s_c3,fea_s_c4),1)
        fea_v_cat=torch.cat((fea_v1,fea_v2,fea_v3,fea_v4),2)

        fea_s_c_r=self.conv_fusing_s_c(fea_s_c_cat)
        fea_v_r=self.conv_fusing_v(fea_v_cat)

        fea_s_c = fea_s_c_r + inter_fea_initial
        fea_v = fea_v_r + intra_fea_initial

        fea_fusion=self.FM(fea_v,fea_s_c)
        fea_fusion=self.Reconstruct(fea_fusion)
        out_res = self.UpSample(fea_fusion)
        out = FormOutput(out_res)+x_upscale
        return out


#调整输出的维度(b, angle*angle, c, h, w)-->(b, 1, H, W)，与输入相对应
def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)
    return out


#特征融合层start
class FM(nn.Module):
    def __init__(self, channel,angRes):
        super(FM, self).__init__()

        self.conv_fusing = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.conv_sharing = nn.Conv2d(angRes * angRes * channel, angRes * angRes * channel, kernel_size=1, stride=1,padding=0)

    def forward(self, intra_fea, inter_fea):
        b, n, c, h, w = intra_fea.shape

        upda_intra_feas = []
        for i in range(n):
            current_sv = intra_fea[:, i, :, :, :].contiguous()
            buffer = torch.cat((current_sv, inter_fea), dim=1)
            buffer = self.lrelu(self.conv_fusing(buffer))
            upda_intra_feas.append(buffer)
        upda_intra_feas = torch.cat(upda_intra_feas, dim=1)
        fuse_fea = self.conv_sharing(upda_intra_feas)
        fuse_fea = fuse_fea.unsqueeze(1).contiguous().view(b,n, c, h, w)

        return fuse_fea

    '''

    class FBM(nn.Module):
    
    Feature Blending
    
    def __init__(self, channel):
        super(FBM, self).__init__()
        self.FERB_1 = RB(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RB(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b*n, c, h, w)
        buffer_1 = self.FERB_1(buffer_init)
        buffer_2 = self.FERB_2(buffer_1)
        buffer_3 = self.FERB_3(buffer_2)
        buffer_4 = self.FERB_4(buffer_3)
        buffer = buffer_4.contiguous().view(b, n, c, h, w)
        return buffer


    '''

class RFDN(nn.Module):
    def __init__(self, channel):
        super(RFDN, self).__init__()

        self.B1 = RFDB(channel)
        self.B2 = RFDB(channel)
        self.B3 = RFDB(channel)
        self.B4 = RFDB(channel)
        self.c  =nn.Conv2d(4*channel,channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.conv=nn.Conv2d(channel,channel,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self, x):
        b, n, c, h, w = x.shape
        input = x.contiguous().view(b * n, c, h, w)

        out_B1 = self.B1(input)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        outB   =self.c(torch.cat([out_B1,out_B2,out_B3,out_B4],dim=1))
        out=self.conv(outB)+input
        output=out.contiguous().view(b, n, -1, h, w)
        return output

class RFDB(nn.Module):
    def __init__(self, in_channels):
        super(RFDB, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, kernel_size=1,stride=1,padding=0)
        self.c1_r = nn.Conv2d(in_channels, self.rc, kernel_size=3,stride=1,padding=1)
        self.c2_d = nn.Conv2d(in_channels, self.dc, kernel_size=1,stride=1,padding=0)
        self.c2_r = nn.Conv2d(in_channels, self.rc, kernel_size=3,stride=1,padding=1)
        self.c3_d = nn.Conv2d(in_channels, self.dc, kernel_size=1,stride=1,padding=0)
        self.c3_r = nn.Conv2d(in_channels, self.rc, kernel_size=3,stride=1,padding=1)
        self.c4 = nn.Conv2d(in_channels, self.dc, kernel_size=3,stride=1,padding=1)
        self.act =  nn.LeakyReLU(0.1, inplace=True)
        self.c5 = nn.Conv2d(self.dc*4, in_channels, kernel_size=1,stride=1,padding=0)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.c5(out)+input

        return out_fused

class fusing_v(nn.Module):
    def __init__(self, channels):
        super(fusing_v, self).__init__()

        self.conv_fusing_v = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b * n, c, h, w)
        buffer = self.conv_fusing_v(buffer_init)
        buffer = buffer.contiguous().view(b, n, -1, h, w)
        return buffer

#特征融合层end


#重建层start
class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out
#重建层end


#特征提取层start
class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = RDASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RDASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x):
        b, n, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        intra_fea_0 = self.FEconv(x)
        intra_fea = self.FERB_1(intra_fea_0)
        intra_fea = self.FERB_2(intra_fea)
        intra_fea = self.FERB_3(intra_fea)
        intra_fea = self.FERB_4(intra_fea)
        _, c, h, w = intra_fea.shape
        intra_fea = intra_fea.unsqueeze(1).contiguous().view(b, n, c, h, w)# intra_fea:  B, N, C, H, W

        return intra_fea

class Extract_inter_fea(nn.Module):
    def __init__(self, channel, angRes):
        super(Extract_inter_fea, self).__init__()
        self.FEconv = nn.Conv2d(angRes*angRes, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = RDASPP(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RDASPP(channel)
        self.FERB_4 = RB(channel)

    def forward(self, x):
        inter_fea_0 = self.FEconv(x)
        inter_fea = self.FERB_1(inter_fea_0)
        inter_fea = self.FERB_2(inter_fea)
        inter_fea = self.FERB_3(inter_fea)
        inter_fea = self.FERB_4(inter_fea)
        return inter_fea


class RDASPPB_Conv(nn.Module):
    def __init__(self, inChannels, outChannels, dilation_rate):
        super(RDASPPB_Conv, self).__init__()
        Cin = inChannels
        Cout = outChannels
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, Cout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(Cout, Cout, kernel_size=3,stride=1,padding=dilation_rate,dilation=dilation_rate,bias=False)

        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDASPP(nn.Module):
    def __init__(self, channel, nConvLayers=3):
        super(RDASPP, self).__init__()
        G0 = channel
        G = channel
        C = nConvLayers
        dilation_rate=[1,2,5]
        convs = []
        for c in range(C):
            convs.append(RDASPPB_Conv(G0 + c * G, G,dilation_rate[c]))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, kernel_size=1, stride=1,padding=0)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x
#特征提取层end

#通道、空间混合注意力模块start
class CoordAtt(nn.Module):
    def __init__(self, inp,reduction = 4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((32, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, 32))
        oup=inp
        mip = max(8, inp//reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.convh = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.convw = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.sigmoid = nn.Sigmoid()

        self.conv4 = RB(oup)
        self.conv5 = RB(oup)

    def forward(self, x):
        b, c, h, w = x.shape
        identity = x
        x_h1 = self.pool_h(x)#(b,c,h,1)
        x_w1 = self.pool_w(x).permute(0, 1, 3, 2)#(b,c,w,1 )

        y = torch.cat([x_h1, x_w1], dim=2)#(b,c,h+w,1 )
        y = self.conv1(y)
        y = self.lrelu(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)#(b,c,1,w )

        x_h = self.sigmoid(self.convh(x_h))#(b,c,h,1 )
        x_w = self.sigmoid(self.convw(x_w))#(b,c,1,w )
        out = identity * x_h * x_w

        out1=self.conv4(out)
        out2=self.conv5(out1)
        out3=out2+x
        output = out3.contiguous().view(b, c, h, w)
        return output


#通道、空间混合注意力模块end

#视图注意力模块start

class VAtt(nn.Module):
    def __init__(self, angle, ratio=4):
        super(VAtt, self).__init__()

        in_channels=angle*angle
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        self.conv1=RB(in_channels)
        self.conv2=RB(in_channels)

    def forward(self, input):
        b, n, c, h, w = input.shape
        angle = int(sqrt(n))
        #[b, n, c, h, w]-->[bc,n, h, w]
        x = input.permute(0,2,1,3,4).contiguous().view(b * c, n, h, w)
        identity = x
        #[bc,n, 1, 1]
        x_GAP=self.avg_pool(x)
        x_MAX=self.max_pool(x)

        avgout=self.sharedMLP(x_GAP)
        maxout=self.sharedMLP(x_MAX)
        out= self.sigmoid(avgout + maxout)*identity
        # [bc,n, h, w]
        out1=self.conv1(out)
        out2=self.conv2(out1)
        out3=x+out2

        output=out3.contiguous().view(b, c, n, h, w).permute(0,2,1,3,4)
        return output

#视图注意力模块end

#调整输入的维度，(b, 1, H, W)-->(b, angle*angle,h, w)
def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])
    data_st = torch.cat(data_sv, dim=1)
    return data_st

def weights_init(m):
    pass

class get_loss(nn.Module):
    def __init__(self,args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, criterion_data=[]):
        loss = self.criterion_Loss(SR, HR)

        return loss


