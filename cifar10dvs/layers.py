import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
from spikingjelly.clock_driven.surrogate import sigmoid

torch.pi = torch.acos(torch.zeros(1)).item() * 2
steps = 16
dt = 5
simwin = dt * steps
a = 0.25
Vth = 0.99999  # 阈值电压 V_threshold
aa = Vth    # 梯度近似项
tau = 0.5  # 漏电常数 tau
conduct = 0.9
linear_decay = Vth/(steps * 2)
print('Vth:',Vth, '\ttau', tau, '\tconduct',conduct, '\tlinear_decay',linear_decay)
# Vth = 0.999  # 阈值电压 V_threshold
# aa = Vth    # 梯度近似项
# tau = 0.5  # 漏电常数 tau
# conduct = 0.9
# linear_decay = Vth/(steps * 2)

DGhard_interval = 1.0

soft_act_gap = 0.02     #（sigmoid之后）           0.015   0.02
soft_gate_act_gap = 0.08 #(过sigmoid之前的数值)      0.06   0.08

gamma_SG = 1.


class SpikeAct_extended(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
        脉冲不可导的地方使用的伪导数
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0.)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()

        # hu is an approximate func of df/du in linear formulation
        hu = abs(input) < 0.5
        hu = hu.float()

        # arctan surrogate function
        # hu =  1 / ((input * torch.pi) ** 2 + 1)

        # triangles
        # hu = (1 / gamma_SG) * (1 / gamma_SG) * ((gamma_SG - input.abs()).clamp(min=0))

        return grad_input * hu,

spikeAct_extended = SpikeAct_extended.apply

class QActF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.round(input)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

QActF = QActF.apply

class QAct(nn.Module):
    def __init__(self, act_bits=2, act_mode="linear", scale_factor=None, zero_point_bias=None):
        super().__init__()
        if scale_factor is not None:
            # self.register_buffer("scale_factor", torch.ones(1)*scale_factor)
            self.scale_factor = scale_factor
        else:
            self.scale_factor = None
        if zero_point_bias is not None:
            # self.register_buffer("scale_factor", torch.ones(1)*scale_factor)
            self.zero_point_bias = zero_point_bias
        else:
            self.zero_point_bias = None
        if act_mode == "linear":
            self.min_int = -2**(act_bits-1)
            self.max_int = 2**(act_bits-1)
        elif act_mode == "gelu":
            self.min_int = -2**(act_bits-1)
            self.max_int = 2**(act_bits-1)
        elif act_mode == "relu":
            self.min_int = 0
            self.max_int = 2**act_bits-1
        else:
            raise ValueError("Unknown 'act_mode'.")
    def __str__(self):
        return "QAct(act_bits=%d, act_mode=%s)" % (self.act_bits, self.act_mode)
    def set_scale(self, scale):
        self.scale_factor = scale
    def set_zero_point(self, zero_point):
        self.zero_point_bias = zero_point
    def forward(self, x):
        if self.scale_factor is None:
            return x, self.scale_factor
        else:
            x = x / self.scale_factor + self.zero_point_bias
            x = torch.clamp(x, self.min_int, self.max_int)
            x = QActF.apply(x)
            x = (x - self.zero_point_bias) * self.scale_factor
            return x, self.scale_factor, self.zero_point_bias



class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to convert.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=Vth):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.register_buffer('Vth', torch.tensor(Vth, dtype=torch.float))

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])   #T*Batch, channel, height, width
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input



class MaxPool_s(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool_s, self).__init__( kernel_size, stride, padding,
                                         dilation, return_indices, ceil_mode)

    def forward(self, input):
        sum = torch.sum(input, dim= -1)
        _, indices= F.max_pool2d(sum, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            return_indices=True)
        indices, _ = torch.broadcast_tensors(indices, torch.zeros((steps,) + indices.shape))
        indices = indices.permute(1, 2, 3, 4, 0)
        out = torch.gather(input.view(input.shape[0], input.shape[1], -1, steps), dim=-2, index=indices.view(sum.shape[0], sum.shape[1], -1, steps)).reshape(indices.shape)
        return out

class LIFSpike_CW(nn.Module):

    def __init__(self, inplace, **kwargs):
        super(LIFSpike_CW, self).__init__()
        self.T = kwargs['t']
        self.soft_mode = kwargs['soft_mode']
        self.static_gate = kwargs['static_gate']
        self.static_param = kwargs['static_param']
        self.time_wise = kwargs['time_wise']
        self.plane = inplace
        #c
        self.alpha = torch.nn.Parameter(torch.tensor(0.2 * (np.random.rand(self.plane) - 0.5)))
        self.beta = torch.nn.Parameter(torch.tensor(0.2 * (np.random.rand(self.plane) - 0.5)))
        self.gamma = torch.nn.Parameter(torch.tensor(0.2 * (np.random.rand(self.plane) - 0.5)))

        self.tau, self.Vth, self.leak = [nn.Parameter(- math.log(1 / i - 1) * torch.ones(self.plane, dtype=torch.float))
                              for i in kwargs['param'][:-1]]
        self.reVth = nn.Parameter(- math.log(1 / kwargs['param'][1] - 1) * torch.ones(self.plane, dtype=torch.float))
        #t, c
        self.conduct = [nn.Parameter(- math.log(1 / i - 1) * torch.ones((self.T, self.plane), dtype=torch.float))
                                   for i in kwargs['param'][3:]][0]

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.T):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau.sigmoid(),
                                                      Vth=self.Vth.sigmoid(),
                                                      leak=self.leak.sigmoid(),
                                                      conduct=self.conduct[step].sigmoid(),
                                                      reVth=self.reVth.sigmoid())
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth, leak, conduct, reVth):
        # print(W_mul_o_t_n1.shape, self.alpha[None, :, None, None].sigmoid().shape)
        if self.static_gate:
            if self.soft_mode:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().sigmoid(), self.beta.view(1, -1, 1, 1).clone().detach().sigmoid(), self.gamma.view(1, -1, 1, 1).clone().detach().sigmoid()
            else:
                al, be, ga = self.alpha.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.beta.view(1, -1, 1, 1).clone().detach().gt(0.).float(), self.gamma.view(1, -1, 1, 1).clone().detach().gt(0.).float()
        else:
            if self.soft_mode:
                # al, be, ga = self.gumbel_continual_gate(self.alpha[None, :, None, None]), self.gumbel_continual_gate(self.beta[None, :, None, None]), self.gumbel_continual_gate(self.gamma[None, :, None, None])
                al, be, ga = self.alpha.view(1, -1, 1, 1).sigmoid(), self.beta.view(1, -1, 1, 1).sigmoid(), self.gamma.view(1, -1, 1, 1).sigmoid()

        # I_t1 = W_mul_o_t_n1 + be * I_t0 * self.conduct.sigmoid()#原先
        I_t1 = W_mul_o_t_n1 * (1 - be * (1 - conduct[None, :, None, None]))
        u_t1_n1 = ((1 - al * (1 - tau[None, :, None, None])) * u_t_n1 * (1 - ga * o_t_n1.clone()) - (1 - al) * leak[None, :, None, None]) + \
                  I_t1 - (1 - ga) * reVth[None, :, None, None] * o_t_n1.clone()
        o_t1_n1 = spikeAct_extended(u_t1_n1 - Vth[None, :, None, None])
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True
        self.tau.copy_(torch.tensor(- math.log(1 / kwargs['param'][0] - 1), dtype=torch.float, device=self.tau.device))
        self.Vth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.Vth.device))
        self.reVth.copy_(torch.tensor(- math.log(1 / kwargs['param'][1] - 1), dtype=torch.float, device=self.reVth.device))

        self.leak.copy_(- math.log(1 / kwargs['param'][2] - 1) * torch.ones(self.T, dtype=torch.float, device=self.leak.device))
        self.conduct.copy_(- math.log(1 / kwargs['param'][3] - 1) * torch.ones(self.T, dtype=torch.float, device=self.conduct.device))

    def gumbel_on(self):
        self.static_gate = False

    def gumbel_off(self):
        self.static_gate = True




class LIFSpike_vanilla(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, plane, tau, Vth):
        super(LIFSpike_vanilla, self).__init__()
        self.tau, self.Vth = torch.tensor(tau), torch.tensor(Vth)

    def forward(self, x): #t, b, c, h, w
        u = torch.zeros(x.shape[1:], device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(int(x.shape[0])):
            u, out[step] = self.extended_state_update(u, out[max(step - 1, 0)], x[step],
                                                      tau=self.tau,
                                                      Vth=self.Vth)
        return out

    #[b, c, h, w]  * [c]
    def extended_state_update(self, u_t_n1, o_t_n1, W_mul_o_t_n1, tau, Vth):
        I_t1 = W_mul_o_t_n1
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1.detach())  + I_t1
        # o_t1_n1 = surrogate_sigmoid.apply(u_t1_n1 - Vth, 4.0)
        o_t1_n1 = spikeAct_extended(u_t1_n1 - Vth)
        return u_t1_n1, o_t1_n1

    def _initialize_params(self, **kwargs):
        self.mid_gate_mode = True

    def gumbel_on(self):
        self.static_gate = False

    def gumbel_off(self):
        self.static_gate = True





