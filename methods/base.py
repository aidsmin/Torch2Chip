"""
Base quantization layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class QBase(nn.Module):
    r"""Base quantization method for weight and activation.

    Args:
    nbit (int): Data precision.
    train_flag (bool): Training mode. 

    Attribute:
    dequantize (bool): Flag for dequantization (int -> descritized float).

    Methods:
    trainFunc (input:Tensor): Training function of quantization-aware training (QAT)
    evalFunc (input:Tensor): Forward pass function of inference. 
    inference(): Switch to inference mode. 
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
        r"""Forward pass of quantization-aware training 
        """
        out = self.q(input)
        return out
    
    def evalFunc(self, input:Tensor):
        r"""Forward pass of inference
        """
        return self.trainFunc(input)
    
    def inference(self):
        r"""Inference mode
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
    r"""Basic low precision convolutional layer

    Inherited from the base nn.Conv2d layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (QBase): Weight quantizer. 
    aq (QBase): Activation quantizer.
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
        Inference mode.
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
    r"""Basic low precision linear layer

    Inherited from the base nn.Linear layer.
    
    Args:
    wbit (int): Weight quantization precision. 
    abit (int): Input quantization precision.
    wq (QBase): Weight quantizer. 
    aq (QBase): Activation quantizer.
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

class CPUQBaseConv2d(QBaseConv2d):
    """
    Low precision layer for CPU-based acceleration
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, wbit: int = 32, abit: int = 32, train_flag=True):
        super(CPUQBaseConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        
        # Define the quantized layer for cpu
        self.qlayer = torch.nn.quantized.Conv2d(self.in_channels, self.out_channels, self.kernel_size, bias=False, 
                            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, dtype=torch.qint32)

    def quint8(self, input:Tensor):
        r"""
        Convert low precision input integer to quint8 data format
        """
        return torch.quantize_per_tensor(input.cpu(), scale=1.0, zero_point=0, dtype=torch.quint8)
    
    def qint8(self, weight:Tensor):
        r"""
        Convert the low precision weight to qint8 data format
        """
        self.qweight = torch.quantize_per_tensor(weight.cpu(), scale=1.0, zero_point=0, dtype=torch.qint8)
        self.b = torch.zeros(self.out_channels)
    
        
    def forward(self, input:Tensor):
        xq = self.aq(input)
        qinput = self.quint8(xq)

        y = torch.nn.quantized.functional.conv2d(qinput, self.qweight, self.b, stride=self.stride, padding=self.padding, scale=1.0, zero_point=0, dtype=torch.qint32)
        # y = self.qlayer(qinput)
        y = torch.dequantize(y)
        return y


class MulShift(nn.Module):
    r"""Multiply the scaling factor and add the bias
    
    Attributes:
    scale: Scaling factor with the shape of output channels.
    bias: Bias value. 
    fl: Fractional bits of the high-precision integer.
    """
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

class MulQuant(nn.Module):
    r"""Multiply the scaling factor and add the bias, then quantize the output.

    Attributes:
    scale: Scaling factor with the shape of output channels.
    bias: Bias value. 
    fl: Fractional bits of the high-precision integer.
    """
    def __init__(self, nbit:int=4):
        super(MulQuant, self).__init__()
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("bias", torch.tensor(0.0))
        self.nbit = nbit
        self.nlv = 2**nbit - 1

        # fractional bit width
        self.fl = 0.

    def forward(self, input:Tensor):
        out = input.mul(self.scale[None, :, None, None]).add(self.bias[None, :, None, None])
        
        # Fused ReLU
        out = F.relu(out)
        out = out.mul(2**(-self.fl)).round()
        
        # quant
        out = out.round()
        out = out.clamp(max=self.nlv)
        
        return out

class ConvBNReLU(nn.Module):
    r"""
    Template of module fusion
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True, int_out=True):
        super(ConvBNReLU, self).__init__()
        
        # modules
        self.conv = QBaseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # scaler and shifter
        if int_out:
            self.scaler = MulQuant(nbit=abit)
        else:
            self.scaler = MulShift()
    
    def forward(self, input:Tensor):
        x = self.conv(input)
        x = self.scaler(x)
        x = self.bn(x)

        x = self.relu(x)
        return x