import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary
from functools import partial
from utils.utils import plot_tensors


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes,  out_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d( out_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x, out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x, out


class CBAM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CBAM, self).__init__()

        self.channel_att = ChannelAttention(in_planes, out_planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out_c, _ = self.channel_att(x)
        out_s, gate_s = self.spatial_att(out_c)
        # plot
        # t_names = ['x','out_c','gate_s','out_s']
        # t_dict = dict()
        # for k in t_names:
        #     t_dict[k] = locals()[k]
        # plot_tensors(t_dict, nrow=2)
        return out_s



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
            updample = F.interpolate(res_block, scale_factor=2, mode='bilinear', align_corners=True)
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


class CBAMResnet18_Unet(nn.Module):

    def __init__(self, n_channels, n_classes, pretrained=False):

        super(CBAMResnet18_Unet, self).__init__()
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

        self.att4 = CBAM(512, 512)
        self.att3 = CBAM(256, 256)
        self.att2 = CBAM(128, 128)
        self.att1 = CBAM(64, 64)
        self.att0 = CBAM(64, 64)

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

        att_block4 = self.att4(encode_block4)
        att_block3 = self.att3(encode_block3)
        decode_block3 = self.conv_decode3(att_block4, att_block3)

        att_block2 = self.att2(encode_block2)
        decode_block2 = self.conv_decode2(decode_block3, att_block2)

        att_block1 = self.att1(encode_block1)
        decode_block1 = self.conv_decode1(decode_block2, att_block1)

        att_block0 = self.att0(encode_block0)
        decode_block0 = self.conv_decode0(decode_block1, att_block0)

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
        att_masks = ['.SpatialGate.sigmoid', '.spatial_att.sigmoid']
        for name, module in self.named_modules():
            for att_mask in att_masks:
                if att_mask in name:
                    hook = partial(self.forward_hook, name=name)
                    self.hook_handles.append(module.register_forward_hook(hook))

    def forward_hook(self, module, input, output, name):
        if type(output) == tuple:
            for i, out in enumerate(output):
                self.feature_maps.append((name+f'.{i}', out.detach().cpu()))
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
    model = CBAMResnet18_Unet(2, 1)
    y = model(x)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis
    f1 = FlopCountAnalysis(model, x)
    print('FLOPs:', f1.total())

    # model = model.cuda()
    # summary(model, (2, 256, 256))
    # model = model.cpu()

    path = 'cbam_resnet18_unet.onnx' # the path of your onnx model
    torch.onnx.export(model, x, path, opset_version=12, input_names=['input'], output_names=['mask'])
    import onnx
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)

    # FLOPs: 24939334144