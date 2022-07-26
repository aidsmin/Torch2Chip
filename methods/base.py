"""
Base quantization layers
"""

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
    
    def trainFunc(self, input):
        """
        Forward pass of quantization-aware training 
        """
        wq = self.wq(self.weight)
        xq = self.aq(input)
        y = F.conv2d(xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y
    
    def evalFunc(self, input):
        """
        Forward pass function for evaluation
        By default, the forward pass of evaluation is equivalent to training
        """
        return self.trainFunc(input)

    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False

    def forward(self, input:Tensor):
        if self.train_flag:
            y = self.trainFunc(input)
        else:
            y = self.evalFunc(input)
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

    def trainFunc(self, input):
        """
        Forward pass of quantization-aware training 
        """
        wq = self.wq(self.weight)
        xq = self.aq(input)
        y = F.linear(xq, wq, self.bias)
        return y
    
    def evalFunc(self, input):
        """
        Forward pass function for evaluation
        By default, the forward pass of evaluation is equivalent to training
        """
        return self.trainFunc(input)
    
    def inference(self):
        """
        Inference mode
        """
        self.train_flag = False
    
    def forward(self, input:Tensor):
        if self.train_flag:
            y = self.trainFunc(input)
        else:
            y = self.evalFunc(input)
        return y

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, 
                padding:int=0, dilation:int=1, groups:int=1, bias:bool=True, wbit:int=32, abit:int=32, train_flag=True):
        super(ConvBNReLU, self).__init__()
        
        # modules
        self.conv = QBaseConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, wbit, abit, train_flag)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, input:Tensor):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x