import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class VanillaUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES,
                 dropout_deep=0.3, dropout_shallow=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024, dropout=dropout_deep)

        self.dec4 = DecoderBlock(1024, 512, 512, dropout=dropout_deep)
        self.dec3 = DecoderBlock(512, 256, 256, dropout=dropout_deep)
        self.dec2 = DecoderBlock(256, 128, 128, dropout=dropout_shallow)
        self.dec1 = DecoderBlock(128, 64, 64, dropout=dropout_shallow)

        self.head = nn.Conv2d(64, num_classes, 1)

        self.apply(_init_weights)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False,
                 dropout_deep=0.3, dropout_shallow=0.1):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.dec5 = DecoderBlock(512, 256, 256, dropout=dropout_deep)
        self.dec4 = DecoderBlock(256, 128, 128, dropout=dropout_deep)
        self.dec3 = DecoderBlock(128, 64, 64, dropout=dropout_shallow)
        self.dec2 = DecoderBlock(64, 64, 64, dropout=dropout_shallow)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
        )

        self.head = nn.Conv2d(32, num_classes, 1)

        self.dec5.apply(_init_weights)
        self.dec4.apply(_init_weights)
        self.dec3.apply(_init_weights)
        self.dec2.apply(_init_weights)
        self.dec1.apply(_init_weights)
        _init_weights(self.head)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.dec5(e5, e4)
        d4 = self.dec4(d5, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2)

        return self.head(d1)


def build_model(variant, pretrained=False):
    if variant == "vanilla":
        return VanillaUNet()
    elif variant == "resnet34":
        return ResNet34UNet(pretrained=pretrained)
    raise ValueError(f"Unknown variant: {variant}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
