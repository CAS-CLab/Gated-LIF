from blocks import *
from layers import *
from spikingjelly.clock_driven import layer
import numpy as np

init_constrain = 0.2

# for CIFAR
class CIFARNet(nn.Module):
    def __init__(self, lif_param: dict, input_size=32, n_class=100, tunable_lif=False):
        super(CIFARNet, self).__init__()
        assert input_size % 32 == 0
        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.arch = [128, 256, "A", 512, "A", 1024, 512]
        self.features = self.make_layer(self.arch, batch_norm=True)
        self.classifier = layer.SeqToANNContainer(
            nn.Linear(512 * 8 * 8, 1024, bias=False),
            # nn.Dropout(p=0.2),
            nn.Linear(1024, 512, bias=False),
            nn.Linear(512, n_class, bias=False),
        )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )


    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.features(x_seq)
        out = torch.flatten(out, 2)
        out = self.classifier(out)
        return out.mean(0)

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                            )
                            )

    def make_layer(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "A":
                layers += [layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))]
            else:
                v = int(v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [layer.SeqToANNContainer(conv2d, tdBatchNorm(v)), LIFSpike_CW(v, **self.lif_param)]
                else:
                    layers += [layer.SeqToANNContainer(conv2d), LIFSpike_CW(v, **self.lif_param)]
                in_channels = v
        return  nn.Sequential(*layers)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResNet_18_stand(nn.Module):
    def __init__(self, lif_param:dict,input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_18_stand, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [2, 2, 2, 2]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [1, 1, 2, 2]
        # self.stage_stride = [2, 2, 1, 1]
        # self.stage_stride = [2, 2, 1, 1]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike(**self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.lastave = layer.SeqToANNContainer(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
            # nn.Linear(256, n_class, bias=False),
            # nn.Linear(n_class, n_class)
        )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )
    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i], nn.Parameter(
                        torch.tensor(init_constrain * (np.random.random() - 0.5), dtype=torch.float)))

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResNet_18_stand_CW(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_18_stand_CW, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [2, 2, 2, 2]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [1, 1, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike_CW(input_channel, **self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock_CW(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        if n_class > 50:
            self.lastave = nn.Sequential(
                nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
                # nn.Linear(256, n_class, bias=False),
                # nn.Linear(n_class, n_class)
            )
        else:
            self.lastave = nn.Sequential(
                nn.Linear(self.stage_out_channels[-1], 256, bias=False),
                nn.Linear(256, n_class, bias=False),
            )


        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )



    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ResNet_18_stand_CW_softsimple(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_18_stand_CW_softsimple, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [2, 2, 2, 2]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [1, 1, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike_CW_softsimple(input_channel, **self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock_CW_softsimple(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.lastave = layer.SeqToANNContainer(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
            # nn.Linear(256, n_class, bias=False),
            # nn.Linear(n_class, n_class)
        )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class ResNet_19_cifar(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_19_cifar, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [3, 3, 2]
        self.stage_out_channels = [-1, 128, 128, 256, 512]
        self.stage_stride = [1, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike(**self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        if n_class > 50:
            self.lastave = layer.SeqToANNContainer(
                nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
                # nn.Linear(256, n_class, bias=False),
                # nn.Linear(n_class, n_class)
            )
        else:
            self.lastave = layer.SeqToANNContainer(
                nn.Linear(self.stage_out_channels[-1], 256, bias=False),
                nn.Linear(256, n_class, bias=False),
            )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i], nn.Parameter(
                        torch.tensor(init_constrain * (np.random.random() - 0.5), dtype=torch.float)))

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class ResNet_19_cifar_CW(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_19_cifar_CW, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [3, 3, 2]
        self.stage_out_channels = [-1, 128, 128, 256, 512]
        self.stage_stride = [1, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike_CW(input_channel, **self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock_CW(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        if n_class > 50:
            self.lastave = layer.SeqToANNContainer(
                nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
                # nn.Linear(256, n_class, bias=False),
                # nn.Linear(n_class, n_class)
            )
        else:
            self.lastave = layer.SeqToANNContainer(
                nn.Linear(self.stage_out_channels[-1], 256, bias=False),
                nn.Linear(256, n_class, bias=False),
            )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)



    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResNet_19_stand_CW_softsimple(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_19_stand_CW_softsimple, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [3, 3, 2]
        self.stage_out_channels = [-1, 128, 128, 256, 512]
        self.stage_stride = [1, 2, 2]



        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                    tdBatchNorm(input_channel),
                                    ),

            LIFSpike_CW_softsimple(input_channel, **self.lif_param),
            )

        # self.spike
        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):  # 循环 4次得到 4个 repeat块组
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            first_block_stride = self.stage_stride[idxstage]

            for i in range(numrepeat):  # 一个repeat块组 组内循环 得到 numrepeat个choice块
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, first_block_stride
                else:
                    inp, outp, stride = input_channel, output_channel, 1

                archIndex += 1  # 计数得到总共有几个 choice block
                print('ResBlock3x3')
                self.features.append(
                    BasicBlock_CW_softsimple(self.lif_param, inp, outp, ksize=3, stride=stride)
                )
                input_channel = output_channel

        self.archLen = archIndex  # 计数得到总共有几个 choice block

        self.globalave = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.lastave = layer.SeqToANNContainer(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False),
            # nn.Linear(256, n_class, bias=False),
            # nn.Linear(n_class, n_class)
        )

        self._initialize_weights()
        print('steps:{}'.format(self.T),
              'init-tau:{}'.format(tau),
              'aa:{}'.format(aa),
              'Vth:{}'.format(Vth)
              )

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i],
                            nn.Parameter(
                                torch.tensor(init_constrain * (np.random.rand(m.plane) - 0.5)
                                             , dtype=torch.float)
                                        )
                            )

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.conv1(x_seq)
        for archs in self.features:
            out = archs(out)
        # print(x.shape)
        out = self.globalave(out)
        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.lastave(out)
        return out.mean(0)


    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, tdBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

