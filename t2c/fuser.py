"""
BatchNorm fusion with full observability
"""

import torch
import copy
import torch.nn as nn
from methods import QBaseConv2d, ConvBNReLU
from methods.base import QBaseLinear

class LayerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        # flag
        self.flag = False
        
        # layers
        self.groups = []
        
        # parameters
        self.xscales = []
        self.xbound = []
    
    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """
        for n, m in model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
        conv_bn_relu = []
        for n, m in self.model.named_modules():
            if isinstance(m, QBaseConv2d):
                self.flag = True
                conv_bn_relu.append(m)

                # scales and boundaries
                self.xscales.append(m.aq.scale.data)
                self.xbound.append(m.aq.alpha.data)
                
                # print("Name: {}, layer: {}".format(n, m))
            
            elif isinstance(m, nn.BatchNorm2d) and self.flag:
                conv_bn_relu.append(m)
                # print("Name: {}, layer: {}".format(n, m))
            
            elif isinstance(m, nn.ReLU) and self.flag:
                conv_bn_relu.append(m)
                # print("Name: {}, layer: {}".format(n, m))
                self.groups.append(conv_bn_relu)
                
                # reset
                self.flag = False
                conv_bn_relu = []

    def fuse(self):
        """
        Fuse conv, bn, and relu layers
        """
        count = 0
        # initialize the model copy to avoid the mutated dict
        fused_model = copy.deepcopy(self.model) 
        
        for name, module in self.model.named_children():
            if len(module) > 0:
                for n, m in module.named_children():
                    if isinstance(m, QBaseConv2d):
                        # fetch the module
                        conv_bn_relu = self.groups[count]
                        bn = conv_bn_relu[1]

                        self.flag = True

                        # fused layer
                        tmp = ConvBNReLU(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, 
                                        wbit=m.wbit, abit=m.abit, train_flag=m.train_flag)

                        # assign modules
                        setattr(tmp, "conv", conv_bn_relu[0])
                        # setattr(tmp, "bn", conv_bn_relu[1])
                        setattr(tmp, "relu", conv_bn_relu[2])

                        # quantization scalers
                        sq = 1 / (tmp.conv.wq.scale.data * tmp.conv.aq.scale.data)

                        # bn scaling
                        std = torch.sqrt(bn.running_var.data + bn.eps)
                        sbn = bn.weight.data.mul(sq) / std
                        # bn bias
                        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std)
                        
                        # scale and bias
                        tmp.scaler.scale.data = sbn
                        tmp.scaler.bias.data = bbn

                        # replace batchnorm by the identity
                        setattr(tmp, "bn", nn.Identity())

                        # update module
                        setattr(module, n, tmp)
                        
                        # increment
                        count += 1
                    elif isinstance(m, nn.BatchNorm2d) and self.flag:
                        tmp = nn.Identity()
                        
                        # replace bn by identity
                        setattr(module, n, tmp)
                    
                    elif isinstance(m, nn.ReLU) and self.flag:
                        tmp = nn.Identity()

                        # replace relu by identity
                        setattr(module, n, tmp)

                        # reset
                        self.flag = False
                        
                setattr(fused_model, name, module)
        
        return fused_model
