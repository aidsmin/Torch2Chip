"""
Base quantization layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class QBase(nn.Module):
    """
    Basic quantizer module
    """
    def __init__(self, nbit:int, train_flag:bool=True):
        super(QBase, self).__init__()
        self.nbit = nbit
        self.train_flag = train_flag
        self.dequantize = True
        self.qflag = True
    
    def q(self, input:Tensor):
        """
        Quantization operation
        """
        return input
    
    def trainFunc(self, input:Tensor):
        """
        Forward pass of quantization-aware training 
        """
        out = self.q(input)
        return out
    
    def evalFunc(self, input:Tensor):
        return self.trainFunc(input)
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
        self.dequantize = False

    def forward(self, input:Tensor):
        if self.train_flag:
            y = self.trainFunc(input)
        else:
            y = self.evalFunc(input)
        return y
    
    def extra_repr(self) -> str:
        return super().extra_repr() + "nbit={}".format(self.nbit)

class QBaseConv2d(nn.Conv2d):
    """
    Basic low precision convolutional layer
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(QBaseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.train_flag = train_flag
        
        self.wbit = wbit
        self.abit = abit
        
        # quantizer
        self.wq = nn.Identity()
        self.aq = nn.Identity()
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
        self.register_buffer("qweight", torch.ones_like(self.weight))
        self.register_buffer("fm_max", torch.tensor(0.))

    def get_fm_info(self, y:Tensor):
        # maximum bit length
        mb = len(bin(int(y.abs().max().item()))) - 2
        fm = mb * y.size(2) * y.size(3)
        
        # maximum featuremap size
        if fm > self.fm_max:
            self.fm_max.data = torch.tensor(fm).float()

    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        
        xq = self.aq(input)
        y = F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # save integer weights
        if not self.train_flag:
            self.qweight.data = wq
            self.get_fm_info(y)

        return y

class QBaseLinear(nn.Linear):
    """
    Basic low precision linear layer
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, wbit:int=32, abit:int=32, train_flag=True):
        super(QBaseLinear, self).__init__(in_features, out_features, bias)
        self.train_flag = train_flag
        
        self.wbit = wbit
        self.abit = abit
        
        # quantizer
        self.wq = nn.Identity()
        self.aq = nn.Identity()
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
    
    def forward(self, input:Tensor):
        wq = self.wq(self.weight)
        xq = self.aq(input)
        y = F.linear(xq, wq, self.bias)
        return y

class MulShift(nn.Module):
    def __init__(self):
        super(MulShift, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))
        
        # fractional bit width
        self.fl = 0.

    def forward(self, input:Tensor):
        out = input.mul(self.scale[None, :, None, None]).add(self.bias[None, :, None, None])
        out = out.mul(2**(-self.fl))
        return out

class ConvBNReLU(nn.Module):
    """
    Template of module fusion
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(ConvBNReLU, self).__init__()
        
        # modules
        self.conv = QBaseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # scaler and shifter
        self.scaler = MulShift()
    
    def forward(self, input:Tensor):
        x = self.conv(input)
        x = self.bn(x)
        x = self.scaler(x)
        x = self.relu(x)
        return x