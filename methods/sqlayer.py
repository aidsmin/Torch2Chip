"""
Sparse low precision layer
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from .base import QBaseConv2d
from .qlayer import SAWB, PACT

class SRSTE(torch.autograd.Function):
    """
    Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch
    
    https://arxiv.org/abs/2102.04010
    """

    @staticmethod
    def forward(ctx, weight, mask, decay = 0.0002):

        ctx.save_for_backward(weight)
        output = weight.clone()
        ctx.mask = mask
        ctx.decay = decay

        return output*mask

    @staticmethod
    def backward(ctx, grad_output):
        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

class SQBaseConv2d(QBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)

        # masks
        self.register_buffer("mask", torch.ones_like(self.weight))
    
    def sparsify(self):
        return self.weight.mul(self.mask)
    
    def forward(self, input: Tensor):
        # sparsify weight
        wq = self.sparsify()
        
        # quantization
        wq = self.wq(wq)
        xq = self.aq(input)
        
        y = F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # save integer weights
        if not self.train_flag:
            self.qweight.data = wq
            self.get_fm_info(y)
        return y
    
class NMConv2d(SQBaseConv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, 
                wbit: int = 32, abit: int = 32, train_flag=True, M:int=4, N:int=2):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        # granularity
        self.M = M
        self.N = N

        # quantizers
        self.wq = SAWB(self.wbit, train_flag=True, qmode="symm")
        self.aq = PACT(self.abit, train_flag=True, alpha=10.0)

    def sparsify(self):
        return SRSTE.apply(self.weight, self.mask)

    def forward(self, input: Tensor):
        return super().forward(input)
