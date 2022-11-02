import torch
import torch.nn as nn

from spikingjelly.clock_driven import layer
from layers import *




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BottleNeck(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, stride=1, wid=None):
        super(BottleNeck, self).__init__()
        assert stride in [1, 2]

        self.lif_param = lif_param
        self.inplanes = inplanes
        self.planes = planes
        width = planes // 2 if wid is None else wid
        self.width = width
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(conv1x1(inplanes, width),
                                            norm_layer(width),
                                            ),
                    LIFSpike(**self.lif_param),
                    layer.SeqToANNContainer(conv3x3(width, width, stride=stride),
                                            norm_layer(width),
                                            ),
                    LIFSpike(**self.lif_param),
                    layer.SeqToANNContainer(conv1x1(width, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))))
        elif inplanes != planes and stride == 1:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))),),
                                            )
        elif inplanes == planes and stride != 1:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(nn.AvgPool2d(2),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))),),
                                            )

        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(nn.AvgPool2d(2),
                                                                    conv1x1(inplanes, planes, stride=1),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))),),
                                            )
        self.stride = stride

        self.lif = LIFSpike(**self.lif_param)

    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(
                        nn.Conv2d(inplanes, planes, kernel_size=ksize, padding=pad, stride=stride, bias=False),
                        norm_layer(planes),
                    ),
                    LIFSpike(**self.lif_param),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))))
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))), ),
                                            )
        self.stride = stride

        self.lif = LIFSpike(**self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out



class BasicBlock_for_imagenet(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock_for_imagenet, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(
                        conv3x3(inplanes, planes, stride=stride),
                        norm_layer(planes),
                    ),
                    LIFSpike(**self.lif_param ),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))),)
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))),
                                                                    ),
                                            )
        self.stride = stride

        self.lif = LIFSpike(**self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out


class BasicBlock_for_imagenet_CW(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock_for_imagenet_CW, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(
                        conv3x3(inplanes, planes, stride=stride),
                        norm_layer(planes),
                    ),
                    LIFSpike_CW(planes, **self.lif_param),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))),)
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))),
                                                                    ),
                                            )
        self.stride = stride

        self.lif = LIFSpike_CW(planes, **self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out

class BasicBlock_CW(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock_CW, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(
                        nn.Conv2d(inplanes, planes, kernel_size=ksize, padding=pad, stride=stride, bias=False),
                        norm_layer(planes),
                    ),
                    LIFSpike_CW(planes, **self.lif_param),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))))
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))), ),
                                            )
        self.stride = stride

        self.lif = LIFSpike_CW(planes, **self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out


class BasicBlock_ann(nn.Module):
    def __init__(self, inplanes, planes, ksize, stride=1):
        super(BasicBlock_ann, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=ksize, padding=pad, stride=stride, bias=False),
                    norm_layer(planes),
                    nn.ReLU(inplace=True),
                    conv3x3(planes, planes),
                    norm_layer(planes),

        )

        if inplanes == planes and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes, stride),
                                            norm_layer(planes),
                                            )

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class BasicBlock_CW_softsimple(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock_CW_softsimple, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    layer.SeqToANNContainer(
                        nn.Conv2d(inplanes, planes, kernel_size=ksize, padding=pad, stride=stride, bias=False),
                        norm_layer(planes),
                    ),
                    LIFSpike_CW_softsimple(planes, **self.lif_param),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))))
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))), ),
                                            )
        self.stride = stride

        self.lif = LIFSpike_CW_softsimple(planes, **self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        out = self.lif(out)
        return out

class BasicBlock_CW_MS(nn.Module):
    def __init__(self, lif_param:dict, inplanes, planes, ksize, stride=1):
        super(BasicBlock_CW_MS, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.lif_param = lif_param
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inplanes = inplanes
        self.planes = planes
        norm_layer = tdBatchNorm

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.body = nn.Sequential(
                    LIFSpike_CW(inplanes, **self.lif_param),
                    layer.SeqToANNContainer(
                        nn.Conv2d(inplanes, planes, kernel_size=ksize, padding=pad, stride=stride, bias=False),
                        norm_layer(planes),
                    ),
                    LIFSpike_CW(planes, **self.lif_param),
                    layer.SeqToANNContainer(conv3x3(planes, planes),
                                            norm_layer(planes, alpha=1/(2 ** (1/2))),
                                            ),
        )

        if inplanes == planes and stride == 1:
            self.downsample = layer.SeqToANNContainer(norm_layer(planes, alpha=1/(2 ** (1/2))))
        else:
            self.downsample = nn.Sequential(layer.SeqToANNContainer(conv1x1(inplanes, planes, stride),
                                                                    norm_layer(planes, alpha=1/(2 ** (1/2))), ),
                                            )
        self.stride = stride

        # self.lif = LIFSpike_CW(planes, **self.lif_param)
    def forward(self, x):
        identity = x
        out = self.body(x)
        identity = self.downsample(identity)
        out += identity
        # out = self.lif(out)
        return out




class SEWBlock_MP(nn.Module):
    """
    This is modified from 'Deep residual learning in spiking neural networks'. We replace their MaxPoolings by AveragePoolings.
    """

    def __init__(self, in_channels, mid_channels, connect_f='ADD', pooling='AP'):
        super(SEWBlock_MP, self).__init__()
        self.connect_f = connect_f
        self.conv =  layer.SeqToANNContainer(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
            )

        if pooling == 'AP':
            self.mp = layer.SeqToANNContainer(
            nn.AvgPool2d(2, 2)
            )
        elif pooling == 'MP':
            self.mp = layer.SeqToANNContainer(
                nn.MaxPool2d(2, 2)
            )
        else:
            self.mp = nn.Sequential()

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out += x
        elif self.connect_f == 'AND':
            out *= x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return self.mp(out)







if __name__=="__main__":
    #(T,B,C,H,W)
    test_data = torch.rand(3, 2, 2, 5, 5)
    model = BasicBlock(2, 4, 3)
    out_data = model(test_data)
    print(out_data.shape)

