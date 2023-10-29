import torch
from torch import nn
import math
import warnings
from model_utils import *
import timm

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class LowFreq(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.sr_ratio = sr_ratio
        if sr_ratio != 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, T, H, W = x.shape
        N = T*H*W
        x_ = x.reshape(B, C, N).permute(0,2,1)
        q = self.q(x_).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio != 1:
            x__ = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x__ = self.norm(x__)
            kv = self.kv(x__).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x.permute(0,2,1).reshape(B, C, T, H, W)

class HighFreq(nn.Module):
    def __init__(self, dim, kernel_size, padding):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x) # B C T H W
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = input + x
        return x

class CA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class UFM(nn.Module):
    def __init__(self, dim, middle, head,sr_ratio):
        super(UFM, self).__init__()
        self.low = LowFreq(dim, head, sr_ratio=sr_ratio)
        self.high = HighFreq(dim, kernel_size=[3,7,7], padding=[1,3,3])
        self.ca = CA(dim)
    def forward(self, x):
        high = self.high(x)
        low = self.low(x)
        ca = self.ca(x)
        return high+low+ca+x

class MultiModalSELayer(nn.Module):
    def __init__(self, channel1, channel2, reduction=16):
        super(MultiModalSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc_modality1 = nn.Sequential(
            nn.Linear(channel1, channel1 // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_modality2 = nn.Sequential(
            nn.Linear(channel2, channel1 // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(nn.Linear(channel1 // reduction*2, channel1, bias=False),
            nn.Sigmoid())

    def forward(self, x1, x2):
        b, c, _, _, _ = x1.size()
        y1 = self.avg_pool(x1).view(b, c)
        y2 = self.avg_pool2(x2).view(b, x2.shape[1])
        y1 = self.fc_modality1(y1)
        y2 = self.fc_modality2(y2)
        y = self.fc2(torch.cat([y1, y2], dim=1)).view(b, c, 1, 1, 1)
        return x1 * y.expand_as(x1)


class MultimodalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, x2):
        B, C, T, H, W = x.shape
        B, C2, _, _ = x2.shape
        N = T*H*W
        x_ = x.reshape(B, C, N).permute(0,2,1)
        q = self.q(x_).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2.reshape(B, C2, -1).permute(0,2,1)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x.permute(0,2,1).reshape(B, C, T, H, W)

class InterM(nn.Module):
    def __init__(self, dim, middle, head,sr_ratio):
        super(InterM, self).__init__()
        self.large = MultimodalAttention(dim, head, sr_ratio=sr_ratio)
        self.se = MultiModalSELayer(dim, dim)
    def forward(self, x, audio):
        large = self.large(x, audio)
        se = self.se(x, audio)
        return se+large+x

class VideoSaliencyModel(nn.Module):
    def __init__(self,
                 transformer_in_channel=32,
                 nhead=4,
                 use_upsample=True,
                 num_hier=3,
                 num_clips=32
                 ):
        super(VideoSaliencyModel, self).__init__()

        self.backbone = BackBoneS3D()
        self.num_hier = num_hier
        print("DecoderConvUP:")
        self.decoder = DecoderConvUp()

    def forward(self, x):
        [y0, y1, y2, y3] = self.backbone(x)
        return self.decoder(y0, y1, y2, y3)
class VideoAudioSaliencyModel(nn.Module):
    def __init__(self,
                 use_transformer=True,
                 transformer_in_channel=768,
                 num_encoder_layers=3,
                 nhead=4,
                 use_upsample=True,
                 num_hier=3,
                 num_clips=32,
                 use_sound = True
                 ):
        super(VideoAudioSaliencyModel, self).__init__()
        self.use_transformer = use_transformer
        self.visual_model = VideoSaliencyModel(
            transformer_in_channel=transformer_in_channel,
            nhead=nhead,
            use_upsample=use_upsample,
            num_hier=num_hier,
            num_clips=num_clips
        )

        if self.use_transformer:
            self.conv_in_1x1 = nn.Conv3d(in_channels=1024, out_channels=transformer_in_channel, kernel_size=1, stride=1,
                                         bias=True)  # b c t h w  1024->576
            self.conv_out_1x1 = nn.Conv3d(in_channels=transformer_in_channel, out_channels=1024, kernel_size=1, stride=1, bias=True)

            self.transformer = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
            self.pool_3d = nn.AdaptiveAvgPool3d((4, 7, 12))
        self.use_sound = use_sound
        if(self.use_sound):
            self.audionet = SoundNet()
            self.audionet.load_state_dict(torch.load('../soundnet8_final.pth'))
            print("Loaded SoundNet Weights")
            for param in self.audionet.parameters():
                param.requires_grad = True

        self.maxpool = nn.MaxPool3d((4, 1, 1), stride=(2, 1, 2), padding=(0, 0, 0))

        # Inter-modality
        self.InterM = InterM(1024, middle=768, head=16, sr_ratio=1)
        self.bilinear = nn.Bilinear(42, 3, 4*7*12)


    def forward(self, x, audio):
        # print("use my model! use_transformer:" , self.use_transformer)
        if(self.use_sound):
            audio = self.audionet(audio)
        [y0, y1, y2, y3] = self.visual_model.backbone(x)
        y_ufm = self.InterM(y0, audio)
        y_bilinear = self.maxpool(y0)
        y_bilinear = self.bilinear(y_bilinear.flatten(2), audio.flatten(2))  # [1,1024,768]
        y_bilinear = y_bilinear.view(y_bilinear.size(0), y_bilinear.size(1), 4, 7, 12)
        y0 = y_ufm + y_bilinear
        if self.use_transformer:
            B, C, T, H, W = y0.shape
            y0 = self.conv_in_1x1(y0)
            y0 = y0.flatten(2).permute(0,2,1)
            cls_tokens = self.transformer.cls_token.expand(B, -1, -1)
            dist_token = self.transformer.dist_token.expand(B, -1, -1)
            fused_out = torch.cat((cls_tokens, dist_token, y0), dim=1)
            for blk in self.transformer.blocks:
                fused_out = blk(fused_out)
            fused_out = self.transformer.norm(fused_out)
            fused_out = fused_out[:, 2:, :].permute(0,2,1).reshape(B, -1, T,H,W)
            y0 = self.conv_out_1x1(fused_out)
        return self.visual_model.decoder(y0, y1, y2, y3)


class DecoderConvUp(nn.Module):
    def __init__(self):
        super(DecoderConvUp, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 832, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(832, 480, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(480, 192, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=(5, 3, 3), stride=(5, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 112 x 192

            nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling,  # 224 x 384

            # 4 time dimension
            nn.Conv3d(32, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )
        self.out2=nn.Sequential(nn.Conv3d(480, 32, kernel_size=(4, 1, 1), stride=(1, 1, 1), bias=False),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear'),
                                nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
                                nn.Sigmoid(),
            )
        self.out3=nn.Sequential(nn.Conv3d(192, 32, kernel_size=(4, 1, 1), stride=(1, 1, 1), bias=False),
                                nn.ReLU(),
                                nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear'),
                                nn.Conv3d(32, 1, kernel_size=1, stride=1, bias=True),
                                nn.Sigmoid(),
                                )

    def forward(self, y0, y1, y2, y3):
        z = self.convtsp1(y0)
        z = torch.cat((z, y1), 2)
        z = self.convtsp2(z)
        out2 = self.out2(z).squeeze(1).squeeze(1)
        z = torch.cat((z, y2), 2)
        z = self.convtsp3(z)
        out3 = self.out3(z).squeeze(1).squeeze(1)
        z = torch.cat((z, y3), 2)
        z = self.convtsp4(z)
        z = z.view(z.size(0), z.size(3), z.size(4))
        if not self.training:
            return z
        return z, out3, out2


class BackBoneS3D(nn.Module):
    def __init__(self):
        super(BackBoneS3D, self).__init__()
        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.base2 = nn.Sequential(
            Mixed_3b(),
            Mixed_3c(),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.base3 = nn.Sequential(
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.base4 = nn.Sequential(
            Mixed_5b(),
            Mixed_5c(),
        )
        self.UFM2 = UFM(480, middle=256, head=10, sr_ratio=[1,4,4])
        self.UFM1 = UFM(832, middle=512, head=13, sr_ratio=[1,2,2])
        self.UFM0 = UFM(1024, middle=768, head=16, sr_ratio=1)

    def forward(self, x):
        y3 = self.base1(x)
        y = self.maxp2(y3)
        y2 = self.base2(y)
        y = self.maxp3(self.UFM2(y2))
        y1 = self.base3(y)
        y = self.maxt4(self.UFM1(y1))
        y = self.maxp4(y)
        y0 = self.base4(y)
        y0 = self.UFM0(y0)
        return [y0, y1, y2, y3]


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)

        return x


if __name__ == '__main__':
    # [1,3,32,224,384] [1,1,70560,1]
    input_video = torch.ones(1, 3, 32, 224, 384)
    input_audio = torch.ones(1, 1, 70560, 1)
    model = VideoAudioSaliencyModel()

    y = model(input_video, input_audio)
    print(y.shape)