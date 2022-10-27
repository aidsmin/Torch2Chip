"""
Customized quantization layers and modules

Example method:
SAWB: Accurate and Efficient 2-bit Quantized Neural Networks
RCF: Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
"""
import torch
import torch.nn as nn
from torch import Tensor
from .q import RCFQuantUQ, STE
from .base import QBaseConv2d, QBaseLinear, QBase

class SAWB(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, qmode:str="symm"):
        super(SAWB, self).__init__(nbit, train_flag)
        self.register_buffer("alpha", torch.tensor(1.0))
        self.register_buffer("scale", torch.tensor(1.0))
        self.qmode = qmode

        # sawb
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        self.z = z_typical[f'{int(nbit)}bit']

    def q(self, input:Tensor):
        """
        Quantization method
        """
        m = input.abs().mean()
        std = input.std()

        if self.qmode == 'symm':
            n_lv = 2 ** (self.nbit - 1) - 1
            self.alpha.data = 1/self.z[0] * std - self.z[1]/self.z[0] * m
        elif self.qmode == 'asymm':
            n_lv = (2 ** (self.nbit) - 1)/2
            self.alpha.data = 2*m
        else:
            raise NotImplemented
    
        self.scale.data = n_lv / self.alpha
        
        if not self.train_flag:
            xq = input.clamp(-self.alpha.item(), self.alpha.item())
            xq = xq.mul(self.scale).round()
            if len(xq.unique()) > 2**self.nbit:
                xq = xq.clamp(-2**self.nbit//2, 2**self.nbit//2-1)
            
            if self.dequantize:
                xq = xq.div(self.scale)
        else:
            xq = input
        return xq

    def trainFunc(self, input:Tensor):
        input = input.clamp(-self.alpha.item(), self.alpha.item())
        # get scaler
        _ = self.q(input)
        # quantization-aware-training
        out = STE.apply(input, self.scale.data)
        return out
    
    def evalFunc(self, input: Tensor):
        out = self.q(input)
        return out

class RCF(QBase):
    def __init__(self, nbit: int, train_flag: bool = True, alpha:float=10.0):
        super(RCF, self).__init__(nbit, train_flag)
        self.register_parameter('alpha', nn.Parameter(torch.tensor(alpha)))
        self.register_buffer("scale", torch.tensor(1.0))

    def q(self, input:Tensor):
        input_q = input.mul(self.scale.data).round()
        input_q = input_q.clamp(max=2**self.nbit-1)

        if self.dequantize:
            input_q = input_q.div(self.scale)
        return input_q

    def trainFunc(self, input:Tensor):
        # pre-computed scaling factor
        nlv = 2**self.nbit - 1
        self.scale = nlv / self.alpha.data

        # quantization-aware-training
        out = RCFQuantUQ.apply(input, self.alpha, self.nbit)
        return out

    def evalFunc(self, input: Tensor):
        if self.qflag:
            # input = input.clamp(max=self.alpha.data)
            input_q = self.q(input)
        else:
            input_q = input
        return input_q

class QConv2d(QBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        
        # quantizers
        self.wq = SAWB(self.wbit, train_flag=True, qmode="symm")
        self.aq = RCF(self.abit, train_flag=True, alpha=10.0)

    def trainFunc(self, input):
        return super().trainFunc(input)

class QLinear(QBaseLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(QLinear, self).__init__(in_features, out_features, bias, wbit, abit, train_flag)

        # quantizers
        self.wq = SAWB(self.wbit, train_flag=True, qmode="symm")
        self.aq = RCF(self.abit, train_flag=True, alpha=10.0)

    def trainFunc(self, input):
        return super().trainFunc(input)