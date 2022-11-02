from blocks import *
from layers import *
from spikingjelly.clock_driven import layer
import numpy as np

init_constrain = 0.2
# for ImageNet
class ResNet_34_stand(nn.Module):
    def __init__(self, lif_param:dict, input_size=224, n_class=1000, tunable_lif=False):
        super(ResNet_34_stand, self).__init__()
        assert input_size % 32 == 0


        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif
        self.stage_repeats = [3, 4, 6, 3]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [2, 2, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=7, stride=2, padding=1, bias=False),
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
                    BasicBlock_for_imagenet(lif_param=self.lif_param, inplanes = inp,planes= outp, ksize=3, stride=stride)
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

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i], nn.Parameter(
                        torch.tensor(init_constrain * (np.random.random() - 0.5), dtype=torch.float)))


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




class ResNet_34_stand_CW(nn.Module):
    def __init__(self, lif_param:dict, input_size=224, n_class=1000, tunable_lif=False):
        super(ResNet_34_stand_CW, self).__init__()
        assert input_size % 32 == 0


        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif
        self.stage_repeats = [3, 4, 6, 3]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [2, 2, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=7, stride=2, padding=1, bias=False),
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
                    BasicBlock_for_imagenet_CW(lif_param=self.lif_param, inplanes = inp,planes= outp, ksize=3, stride=stride)
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

    def randomize_gate(self):
        for name, m in self.named_modules():
            if all([hasattr(m, i) for i in self.choice_param_name]):
                for i in range(len(self.choice_param_name)):
                    setattr(m, self.choice_param_name[i], nn.Parameter(
                        torch.tensor(init_constrain * (np.random.random() - 0.5), dtype=torch.float)))

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


class ResNet_18_stand_CW_MS(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_18_stand_CW_MS, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [2, 2, 2, 2]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [2, 2, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=7, stride=2, padding=1, bias=False),
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
                    BasicBlock_CW_MS(self.lif_param, inp, outp, ksize=3, stride=stride)
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



class ResNet_34_stand_CW_MS(nn.Module):
    def __init__(self, lif_param:dict, input_size=32, n_class=100, tunable_lif=False):
        super(ResNet_34_stand_CW_MS, self).__init__()
        assert input_size % 32 == 0

        self.choice_param_name = ['alpha', 'beta', 'gamma']
        self.lifcal_param_name = ['tau', 'Vth', 'leak', 'conduct', 'reVth']
        self.T = lif_param['t']
        self.lif_param = lif_param
        self.tunable_lif = tunable_lif

        self.stage_repeats = [3, 4, 6, 3]
        self.stage_out_channels = [-1, 64, 64, 128, 256, 512]
        self.stage_stride = [2, 2, 2, 2]
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(nn.Conv2d(3, input_channel, kernel_size=7, stride=2, padding=1, bias=False),
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
                    BasicBlock_CW_MS(self.lif_param, inp, outp, ksize=3, stride=stride)
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


