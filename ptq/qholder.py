"""
Quantized model holder for Post Training Quantization
"""

import torch
import torch.nn as nn

class QModel(nn.Module):
    def __init__(self, model):
        super(QModel, self).__init__()
        
        # quant and dequant 
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x