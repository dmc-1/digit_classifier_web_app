import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MiniResNet(nn.Module):
    def __init__(self, config):
        super(MiniResNet, self).__init__()

        backbone_config = config["backbone"]
        head_config = config["head"]
        head_config["fc_input_dim"] = backbone_config["res_blocks"][-1]["out_channels"]

        self.backbone = self._build_backbone(backbone_config)
        self.head = self._build_head(head_config)

    def _build_backbone(self, backbone_config):
        layers = []

        input_channels = backbone_config['input_channels']
        in_channels = backbone_config['initial_channels']
        conv1 = nn.Conv2d(input_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        layers.append(conv1)
        bn1 = nn.BatchNorm2d(in_channels)
        layers.append(bn1)
        layers.append(nn.ReLU())

        for res_block_config in backbone_config['res_blocks']:
            layers.append(ResBlock(in_channels, res_block_config['out_channels'], res_block_config['stride']))
            in_channels = res_block_config['out_channels']

        return nn.Sequential(*layers)

    def _build_head(self, head_config):
        layers = []

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())

        fc = nn.Linear(head_config['fc_input_dim'], head_config['num_classes'])
        layers.append(fc)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out


def build_model(model_config):
    return MiniResNet(model_config)