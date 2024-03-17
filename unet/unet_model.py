""" Full assembly of the parts to form the complete network """

from functools import partial
from .unet_parts import *
from torchsummary import summary


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.init_weights()

        self.hook_handles = []
        self.feature_maps = []

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
                torch.nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
    
    def register_forward_hooks(self):
        for name, module in self.named_children():
            hook = partial(self.forward_hook, name=name)
            self.hook_handles.append(module.register_forward_hook(hook))
        layers = ['up1.up', 'up2.up', 'up3.up', 'up4.up']
        for name, module in self.named_modules():
            if name in layers:
                hook = partial(self.forward_hook, name=name)
                self.hook_handles.append(module.register_forward_hook(hook))

    def forward_hook(self, module, input, output, name):
        self.feature_maps.append((name, output.detach().cpu()))

    def close_forward_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def clear_feature_maps(self):
        self.feature_maps.clear()


if __name__ == '__main__':
    x = torch.rand(1, 2, 256, 256)
    model = UNet(2, 1, bilinear=True)
    y = model(x)
    print(y.shape)

    from fvcore.nn import FlopCountAnalysis
    f1 = FlopCountAnalysis(model, x)
    print('FLOPs:', f1.total())

    model = model.cuda()
    summary(model, (2, 256, 256))
    model = model.cpu()

    path = 'unet.onnx' # the path of your onnx model
    torch.onnx.export(model, x, path, opset_version=12, input_names=['input'], output_names=['mask'])
    import onnx
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)

    # FLOPs: 40123236352    # bilinear=True