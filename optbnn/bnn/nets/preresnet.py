import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.conv import Conv2d
from ..layers.linear import Linear


def conv3x3(in_planes, out_planes, stride=1, scaled_variance=True):
    "3x3 convolution with padding"
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False, scaled_variance=scaled_variance)


def init_norm_layer(inplanes, norm_layer, **kwargs):
    assert norm_layer in ['batchnorm',  None]
    if norm_layer == 'batchnorm':
        return nn.BatchNorm2d(inplanes, eps=0, momentum=None, affine=False,
                              track_running_stats=False)
    elif norm_layer is None:
        return nn.Identity()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer='batchnorm', scaled_variance=True):
        super(BasicBlock, self).__init__()
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride,
                             scaled_variance=scaled_variance)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        self.conv2 = conv3x3(planes, planes,
                             scaled_variance=scaled_variance)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer='batchnorm', scaled_variance=True):
        super(Bottleneck, self).__init__()
        self.norm1 = init_norm_layer(inplanes, norm_layer)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False,
                            scaled_variance=scaled_variance)
        self.norm2 = init_norm_layer(inplanes, norm_layer)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False,
                            scaled_variance=scaled_variance)
        self.norm3 = init_norm_layer(inplanes, norm_layer)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False,
                            scaled_variance=scaled_variance)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):
    def __init__(self, depth, num_classes=10, block_name='BasicBlock',
                 norm_layer='batchnorm', scaled_variance=True):
        super(PreResNet, self).__init__()
        self.scaled_variance = scaled_variance
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, padding=1,
                            bias=False, scaled_variance=scaled_variance)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.norm = init_norm_layer(64 * block.expansion, norm_layer)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = Linear(64 * block.expansion, num_classes,
                         scaled_variance=scaled_variance)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear) or isinstance(m, Conv2d):
                m.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False,
                       scaled_variance=self.scaled_variance),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            scaled_variance=self.scaled_variance))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                scaled_variance=self.scaled_variance))

        return nn.Sequential(*layers)

    def forward(self, x, log_softmax=False):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.norm(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if log_softmax:
            x = F.log_softmax(x, dim=1)

        return x

    def predict(self, x):
        self.eval()
        predictions = self.forward(x, log_softmax=True)

        predictions = torch.exp(predictions)

        return predictions

