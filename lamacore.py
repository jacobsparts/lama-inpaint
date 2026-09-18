"""Vendored LaMa (big-lama) FFC ResNet generator - standalone port.

Architecture and forward semantics are ported from the original
saicinpainting reference implementation (training/modules/ffc.py) so that the
parameter names layout matches big-lama's checkpoint exactly
(989 keys / 51,057,179 params: generator.model.{1..34}).

Only torch is required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(kind="relu"):
    if kind is None:
        return nn.Identity()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "relu_uc":
        return nn.ReLU(inplace=False)
    if kind == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if kind == "silu":
        return nn.SiLU(inplace=True)
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation {kind}")


# ---------------------------------------------------------------- FFC bits


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None,
                 spatial_scale_mode="bilinear", spectral_pos_encoding=False, use_gca=False,
                 use_ffc_dropout=False, ffc_dropout_input=0.0, ffc_dropout_output=0.0,
                 ffc_dropout_type="dropout"):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2, out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.ffc_dropout = None

        if spatial_scale_factor or spatial_scale_mode is not None:
            self.ffc_scale = nn.Upsample(scale_factor=spatial_scale_factor, mode=spatial_scale_mode)
        else:
            self.ffc_scale = nn.Identity()

    def forward(self, x):
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (b, c, 2, h, w)
        ffted = ffted.view(ffted.shape[0], -1, ffted.shape[-2], ffted.shape[-1])
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view(ffted.shape[0], ffted.shape[1] // 2, 2,
                           ffted.shape[-2], ffted.shape[-1])
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        out = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim, norm="ortho")
        return out


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super().__init__()
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        xs = 0
        x = self.conv2(x + output + xs)
        return x


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin=0, ratio_gout=0, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, enable_lfu=True, padding_type="reflect",
                 gated=False, **spectral_kwargs):
        super().__init__()
        assert stride == 1 or stride == 2
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2,
                              enable_lfu, **spectral_kwargs)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin=0, ratio_gout=0, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, enable_lfu=True, **kwargs):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout, stride,
                       padding, dilation, groups, bias, enable_lfu, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True) if lact is not nn.Identity else lact()
        self.act_g = gact(inplace=True) if gact is not nn.Identity else gact()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.inline = inline
        self.concat = ConcatTupleLayer()

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        return x_l, x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, (tuple, list))
        x_l, x_g = x
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat((x_l, x_g), dim=1)


# ------------------------------------------------------------ the generator


class FFCResNetGenerator(nn.Module):
    """big-lama generator: input_nc 4 (RGB + mask), output_nc 3, ngf 64,
    n_downsampling 3, n_blocks 18 (-> 9 resblocks), ratio_gout 0.75 in the
    bottleneck, sigmoid output. Module layout matches the reference
    saicinpainting generator exactly (ReflectionPad2d at index 0 and 33)."""

    def __init__(self, input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18,
                 norm_layer=nn.BatchNorm2d, padding_type="reflect", activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs=None, downsample_conv_kwargs=None, resnet_conv_kwargs=None,
                 add_out_act="sigmoid", max_features=1024):
        super().__init__()
        # big-lama configuration (config.yaml): the initial conv and every
        # downsample stage are fully local (ratio 0) except the last, whose
        # ratio_gout follows the bottleneck's ratio_gin; the bottleneck
        # resblocks are mixed 0.75 local / 0.25 global.
        init_conv_kwargs = init_conv_kwargs if init_conv_kwargs is not None else {}
        downsample_conv_kwargs = (
            downsample_conv_kwargs if downsample_conv_kwargs is not None else {})
        resnet_conv_kwargs = (
            resnet_conv_kwargs if resnet_conv_kwargs is not None
            else {"ratio_gin": 0.75, "ratio_gout": 0.75})
        self.add_out_act = add_out_act

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        for i in range(n_blocks):
            model += [FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                     activation_layer=activation_layer, norm_layer=norm_layer,
                                     **resnet_conv_kwargs)]

        model += [ConcatTupleLayer()]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation("tanh" if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        if input.dim() == 3:
            input = input.unsqueeze(0)
        return self.model(input)
