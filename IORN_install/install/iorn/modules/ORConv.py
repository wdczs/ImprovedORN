import math
import torch
import torch.nn.functional as F
from torch.nn.modules import Conv2d
from torch.nn.modules.utils import _pair
from ..functions import mapping_rotate
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd.variable import Variable


class ORConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, arf_config, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.nOrientation, self.nRotation = _pair(arf_config)
        assert (math.log(self.nOrientation) + 1e-5) % math.log(2) < 1e-3, 'invalid nOrientation {}'.format(self.nOrientation)
        assert (math.log(self.nRotation) + 1e-5) % math.log(2) < 1e-3, 'invalid nRotation {}'.format(self.nRotation)

        super(ORConv2d, self).__init__(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups, bias)
        self.register_buffer("indices", self.get_indices())
        # re-create weight/bias
        self.weight.data.resize_(out_channels, in_channels, self.nOrientation, *self.kernel_size)
        if bias:
            self.bias.data.resize_(out_channels * self.nRotation)
        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_channels * self.nOrientation
        n = self.out_channels * self.nRotation
        for k in self.kernel_size:
            n *= k
        self.weight.data.normal_(0, math.sqrt(2.0 / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def get_indices(self, mode='fast'):
        kernel_indices = {
            1: {
                0: (1,),
                45: (1,),
                90: (1,),
                135: (1,),
                180: (1,),
                225: (1,),
                270: (1,),
                315: (1,)
            },
            3: {
                0: (1,2,3,4,5,6,7,8,9),
                45: (2,3,6,1,5,9,4,7,8),
                90: (3,6,9,2,5,8,1,4,7),
                135: (6,9,8,3,5,7,2,1,4),
                180: (9,8,7,6,5,4,3,2,1),
                225: (8,7,4,9,5,1,6,3,2),
                270: (7,4,1,8,5,2,9,6,3),
                315: (4,1,2,7,5,3,8,9,6)
            }
        }
        delta_orientation = 360 / self.nOrientation
        delta_rotation = 360 / self.nRotation
        kH, kW = self.kernel_size
        indices = torch.ByteTensor(self.nOrientation * kH * kW, self.nRotation)
        for i in range(0, self.nOrientation):
            for j in range(0, kH * kW):
                for k in range(0, self.nRotation):
                    angle = delta_rotation * k
                    layer = (i + math.floor(angle / delta_orientation)) % self.nOrientation
                    kernel = kernel_indices[kW][angle][j]
                    indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
        return indices.view(self.nOrientation, kH, kW, self.nRotation)

    def rotate_arf(self):
        return mapping_rotate(self.weight, self.indices)

    def forward(self, input):
        return F.conv2d(input, self.rotate_arf(), self.bias, self.stride,
                self.padding, self.dilation, self.groups)

    def __repr__(self):
        arf_config = '[{}]'.format(self.nOrientation) \
            if self.nOrientation == self.nRotation \
            else '[{}-{}]'.format(self.nOrientation, self.nRotation)
        s = ('{name}({arf_config} {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, arf_config=arf_config, **self.__dict__)


class ORBatchNorm2d(Module):

    def __init__(self, num_features, nOrientation, eps=1e-5, momentum=0.1, affine=True):
        super(ORBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.nOrientation = nOrientation
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        batch_size, channels, h, w = input.size()

        input_reshaped = input.view(batch_size, channels//self.nOrientation, h*self.nOrientation, w)

        result = F.batch_norm(
            input_reshaped, self.running_mean, self.running_var, self.weight, self.bias,
            self.training, self.momentum, self.eps)

        return result.view(batch_size, channels, h, w)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))