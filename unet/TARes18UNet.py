import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from .triplet_attention import TripletAttention
from functools import partial
from utils.utils import plot_tensors


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            # nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            nn.BatchNorm2d(out_planes)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self, in_compress_channel=0, kernel_size=7):
        super(SpatialGate, self).__init__()
        if in_compress_channel == 0:
            self.spatial = BasicConv(
                2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
            )
        else:
            self.in_conv = BasicConv(
                in_compress_channel, 2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=True
            )
            self.spatial = BasicConv(
                4, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
            )
        self.compress = ChannelPool()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, in_compress=None):
        x_compress = self.compress(x)
        if in_compress is not None:
            # in_compress = F.interpolate(in_compress, scale_factor=2, mode='bilinear', align_corners=True)
            in_compress = F.interpolate(in_compress, size=x_compress.shape[-2:], mode='bilinear', align_corners=True)
            in_compress = self.in_conv(in_compress)
            x_compress = torch.cat([x_compress, in_compress], dim=1)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        # return x * scale, None
        return x * scale, x_compress


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
        depth=-1,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            if depth == -1:
                self.SpatialGate = SpatialGate()
            elif depth == -2:
                self.SpatialGate = SpatialGate(in_compress_channel=2)
            else:
                self.SpatialGate = SpatialGate(in_compress_channel=4)

    def forward(self, x, in_compress=None):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()    # N,H,C,W
        x_out1, _ = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()    # N,W,H,C
        x_out2, _ = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out31, out_compress = self.SpatialGate(x, in_compress)
            x_out = (1 / 3) * (x_out31 + x_out11 + x_out21)
        else:
            out_compress = None
            x_out = (1 / 2) * (x_out11 + x_out21)
        # plot
        # t_names = ['x','x_out11','x_out21','x_out31','in_compress','out_compress','x_out']
        # t_dict = dict()
        # for k in t_names:
        #     t_dict[k] = locals()[k]
        # plot_tensors(t_dict, nrow=2)
        return x_out, out_compress


class expansive_block(nn.Module):
    def __init__(self, res_in_channels, res_mid_channels, res_out_channels):
        super(expansive_block, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=res_in_channels, out_channels=res_mid_channels, padding=1),
            nn.BatchNorm2d(res_mid_channels),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=res_mid_channels, out_channels=res_out_channels, padding=1),
            nn.BatchNorm2d(res_out_channels),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(kernel_size=(1, 1), in_channels=res_in_channels, out_channels=res_out_channels),
            nn.BatchNorm2d(res_out_channels),
        )
        self.after_add = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, down, skip=None):
        out1 = self.block1(down)
        out2 = self.block2(down)
        res_block = self.after_add(torch.add(out1, out2))
        if skip is not None:
            # updample = F.interpolate(res_block, scale_factor=2, mode='bilinear', align_corners=True)
            updample = F.interpolate(res_block, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            out = torch.cat([updample, skip], dim=1)
            return out
        else:
            return res_block


def final_block(in_channels, mid_channels, out_channels):
    block = nn.Sequential(
        expansive_block(in_channels, mid_channels, mid_channels),
        nn.Conv2d(kernel_size=(1, 1), in_channels=mid_channels, out_channels=out_channels),
        nn.BatchNorm2d(out_channels),
    )
    return block


class TAResnet18_Unet(nn.Module):

    def __init__(self, n_channels, n_classes, pretrained=False):

        super(TAResnet18_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True

        self.resnet = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),  # downsample
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # downsample
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.layer1 = nn.Sequential(
            self.resnet.maxpool,    # downsample
            self.resnet.layer1      # conv*4
        )
        # self.layer1 = self.resnet.layer1    # conv*4
        self.layer2 = self.resnet.layer2    # downsample + conv*4
        self.layer3 = self.resnet.layer3    # downsample + conv*4
        self.layer4 = self.resnet.layer4    # downsample + conv*4

        self.conv_decode3 = expansive_block(512, 256, 256)  # upsample + cat + conv*2
        self.conv_decode2 = expansive_block(256+256, 256, 128)  # upsample + cat + conv*2
        self.conv_decode1 = expansive_block(128+128, 128, 64)     # upsample + cat + conv*2
        self.conv_decode0 = expansive_block(64+64, 64, 64)         # upsample + cat + conv*2

        self.ta4 = TripletAttention(None, depth=-1)
        self.ta3 = TripletAttention(None, depth=-2)
        self.ta2 = TripletAttention(None, depth=-3)
        self.ta1 = TripletAttention(None, depth=-4)
        self.ta0 = TripletAttention(None, depth=-5)

        self.final_layer = final_block(128, 64, n_classes)   # conv*1

        self.init_weights()

        self.hook_handles = []
        self.feature_maps = []

    def forward(self, x):
        encode_block0 = self.layer0(x)
        encode_block1 = self.layer1(encode_block0)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        ta_block4, out_compress4 = self.ta4(encode_block4)
        ta_block3, out_compress3 = self.ta3(encode_block3, out_compress4)
        decode_block3 = self.conv_decode3(ta_block4, ta_block3)

        ta_block2, out_compress2 = self.ta2(encode_block2, out_compress3)
        decode_block2 = self.conv_decode2(decode_block3, ta_block2)

        ta_block1, out_compress1 = self.ta1(encode_block1, out_compress2)
        decode_block1 = self.conv_decode1(decode_block2, ta_block1)

        ta_block0, _ = self.ta0(encode_block0, out_compress1)
        decode_block0 = self.conv_decode0(decode_block1, ta_block0)

        final_layer = self.final_layer(decode_block0)

        return final_layer

    def init_weights(self):
        for name, mod in self.named_modules():
            if type(mod) in [nn.Conv2d]:
                torch.nn.init.xavier_normal_(mod.weight.data, gain=1)
                if mod.bias is not None:
                    torch.nn.init.constant_(mod.bias.data, 0.0)
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
    
    # def train(self, mode: bool = True):
    #     self.resnet.train(mode)
    #     return super().train(mode)

    # def eval(self):
    #     self.resnet.eval()
    #     return super().eval()

    def register_forward_hooks(self):
        for name, module in self.named_children():
            hook = partial(self.forward_hook, name=name)
            self.hook_handles.append(module.register_forward_hook(hook))
        att_mask = '.SpatialGate.sigmoid'
        for name, module in self.named_modules():
            if att_mask in name:
                hook = partial(self.forward_hook, name=name)
                self.hook_handles.append(module.register_forward_hook(hook))

    def forward_hook(self, module, input, output, name):
        if type(output) == tuple:
            for i, out in enumerate(output):
                self.feature_maps.append((name+f'.out{i}', out.detach().cpu()))
        elif type(output) == torch.Tensor:
            self.feature_maps.append((name, output.detach().cpu()))
        else:
            raise Exception(f'error in hook of <torch.nn.Module>: {name}')

    def close_forward_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def clear_feature_maps(self):
        self.feature_maps.clear()


if __name__ == '__main__':
    x = torch.rand(1, 2, 256, 256)
    model = TAResnet18_Unet(2, 1)
    y = model(x)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis
    f1 = FlopCountAnalysis(model, x)
    print('FLOPs:', f1.total())

    # model = model.cuda()
    # summary(model, (2, 256, 256))
    # model = model.cpu()

    path = 'ta_resnet18_unet4.onnx' # the path of your onnx model
    torch.onnx.export(model, x, path, opset_version=12, input_names=['input'], output_names=['mask'])
    import onnx
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)

    # FLOPs: 24942326528    # no connections between TAMs
    # FLOPs: 24987030272    # with connections between TAMs