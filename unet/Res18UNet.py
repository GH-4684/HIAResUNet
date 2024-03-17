import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary


class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
        )
        self.block2 = nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels)
        self.after_add = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out1 = self.block1(cat)
            out2 = self.block2(cat)
            out = torch.add(out1, out2)
            out = self.after_add(out)
        # else:
        #     out = self.block(d)
        #     out = torch.add(d, out)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet18_Unet(nn.Module):

    def __init__(self, n_channels, n_classes, pretrained=False):

        super(Resnet18_Unet, self).__init__()
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

        # self.layer1 = self.resnet.layer1    # conv*4
        self.layer1 = nn.Sequential(
            self.resnet.maxpool,    # downsample
            self.resnet.layer1      # conv*4
        )
        self.layer2 = self.resnet.layer2    # downsample + conv*4
        self.layer3 = self.resnet.layer3    # downsample + conv*4
        self.layer4 = self.resnet.layer4    # downsample + conv*4

        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64+64, 32, 32)
        self.final_layer = final_block(32, n_classes)

    def forward(self, x):
        encode_block0 = self.layer0(x)
        encode_block1 = self.layer1(encode_block0)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        decode_block3 = self.conv_decode3(encode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1, encode_block0)
        final_layer = self.final_layer(decode_block0)

        return final_layer


if __name__ == '__main__':
    x = torch.rand(1, 2, 256, 256)
    model = Resnet18_Unet(2, 1)
    y = model(x)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis
    f1 = FlopCountAnalysis(model, x)
    print('FLOPs:', f1.total())

    model = model.cuda()
    summary(model, (2, 256, 256))
    model = model.cpu()

    path = 'resnet18_unet.onnx' # the path of your onnx model
    torch.onnx.export(model, x, path, opset_version=12, input_names=['input'], output_names=['mask'])
    import onnx
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)

    # FLOPs: 22665691136