"""
BatchNorm fusion with full observability
"""

import torch
import copy
import torch.nn as nn
from methods import QBaseConv2d, ConvBNReLU
from typing import List

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

        # full precision conv layer
        self.fpl = 0

        # full precision classifier
        self.fpc = False
    
    def inference(self):
        """
        Switch to inference mode
        """
        for n, m in self.model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def layers(self):
        """
        Fetch layer information from pretrained model
        """
        conv_bn_relu = []
        l = 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, "wbit"):
                self.fpl += 1
            
            elif isinstance(m, QBaseConv2d):
                self.flag = True
                conv_bn_relu.append(m)

                # scales and boundaries
                self.xscales.append(m.aq.scale.data)
                self.xbound.append(m.aq.alpha.data)
                l += 1
            
            elif isinstance(m, nn.BatchNorm2d) and self.flag:
                conv_bn_relu.append(m)
                print("Name: {}, layer: {}".format(n, m))
            
            elif isinstance(m, nn.ReLU) and self.flag:
                conv_bn_relu.append(m)
                # print("Name: {}, layer: {}".format(n, m))
                self.groups.append(conv_bn_relu)
                
                # reset
                self.flag = False
                conv_bn_relu = []
            
            elif isinstance(m, nn.Linear) and not hasattr(m, "wbit"):
                self.fpc = True

    def conv_bn_relu(self, cbr:List, l=-1.0, snxt:float=1.0, int_out:bool=False):
        assert len(cbr) == 3, "The input must include conv, bn, and relu modules"
        conv, bn, _ = cbr

        # fused layer
        tmp = ConvBNReLU(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                        wbit=conv.wbit, abit=conv.abit, train_flag=conv.train_flag, int_out=int_out)
        
        # assign modules
        setattr(tmp, "conv", cbr[0])
        setattr(tmp, "bn", cbr[1])
        setattr(tmp, "relu", cbr[2])

        # quantization scalers
        sq = 1 / (tmp.conv.wq.scale.data * tmp.conv.aq.scale.data)

        # bn scaling
        std = torch.sqrt(bn.running_var.data + bn.eps)
        sbn = bn.weight.data.mul(sq) / std
        # bn bias
        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std)
        
        # scale and bias
        tmp.scaler.scale.data = sbn.mul(snxt)
        tmp.scaler.bias.data = bbn.mul(snxt)

        # replace batchnorm by the identity
        setattr(tmp, "bn", nn.Identity())

        # replace the activation quantizer by the Identity module
        if l > self.fpl-1:
            tmp.conv.aq = nn.Identity()
        
        return tmp

    def fuse(self):
        """
        Fuse conv, bn, and relu layers
        """
        l = 0   # layer counter
        # initialize the model copy to avoid the mutated dict
        fused_model = copy.deepcopy(self.model) 
        
        for name, module in self.model.named_children():
            if len(module) > 0:
                for n, m in module.named_children():
                    if isinstance(m, QBaseConv2d):
                        # fetch the module
                        conv_bn_relu = self.groups[l]
                        bn = conv_bn_relu[1]

                        self.flag = True

                        if l < len(self.xscales)-1:
                            snxt = self.xscales[l+1]
                            int_out = True
                        else:
                            snxt = 1.0
                            if self.fpc:
                                int_out = False

                        # fused layer
                        tmp = ConvBNReLU(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, 
                                        wbit=m.wbit, abit=m.abit, train_flag=m.train_flag, int_out=int_out)

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
                        tmp.scaler.scale.data = sbn.mul(snxt)
                        tmp.scaler.bias.data = bbn.mul(snxt)

                        # replace batchnorm by the identity
                        setattr(tmp, "bn", nn.Identity())

                        # replace the activation quantizer by the Identity module
                        if l > self.fpl-1:
                            tmp.conv.aq = nn.Identity()

                        # update module
                        setattr(module, n, tmp)
                        
                        # increment
                        l += 1
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


class MobileNetFuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super(MobileNetFuser, self).__init__(model)
    
    def fuse(self):
        """
        Fuse conv, layer, relu for MobileNet architecture
        """
        l = 0   # layer counter
        # initialize the model copy to avoid the mutated dict
        fused_model = copy.deepcopy(self.model) 

        for name, module in self.model.named_children():
            if isinstance(module, (nn.AvgPool2d, nn.Linear)):
                continue
            else:
                # layers in the bottom level sequential
                for n, m in module.named_children():
                    assert len(m) > 0
                    seq = []
                    for layer in m.modules():
                        if isinstance(layer, nn.Conv2d) and not hasattr(layer, "wbit"):
                            seq.append(layer)

                        elif isinstance(layer, QBaseConv2d):
                            # fetch the module
                            conv_bn_relu = self.groups[l]
                            self.flag = True

                            if l < len(self.xscales)-1:
                                snxt = self.xscales[l+1]
                                int_out = True
                            else:
                                snxt = 1.0
                                if self.fpc:
                                    int_out = False

                            tmp = self.conv_bn_relu(conv_bn_relu, l=l, snxt=snxt, int_out=int_out)
                            
                            l += 1
                            seq.append(tmp)

                        elif isinstance(layer, nn.BatchNorm2d):
                            if l != 0:
                                tmp = nn.Identity()
                                seq.append(tmp)
                            else:
                                seq.append(layer)
                        
                        elif isinstance(layer, nn.ReLU):
                            if l != 0:
                                tmp = nn.Identity()
                                seq.append(tmp)
                            else:
                                seq.append(layer)
                            self.flag = False
                    
                    # reconstruct    
                    seq = nn.Sequential(*seq)
                    setattr(module, n, seq)
                setattr(fused_model, name, module)
    
        return fused_model

            
class ResNetFuser(LayerFuser):
    def __init__(self, model: nn.Module):
        super().__init__(model)
        
    def layers(self):
        pass

    def fuse(self):
        for name, module in self.model.named_children():
            if "layer" in name:
                for basic_block_name, basic_block in module.named_children():
                    cbr = [basic_block.conv1, basic_block.bn1, basic_block.relu1]
                    cb = [basic_block.conv2, basic_block.bn2, nn.Identity()]
                    
                    # get fused modules
                    fm1 = self.conv_bn_relu(cbr)
                    fm2 = self.conv_bn_relu(cb)

                    # update modules
                    basic_block.conv1 = fm1
                    basic_block.conv2 = fm2 

                    # disable other modules
                    basic_block.bn1 = nn.Identity()
                    basic_block.bn2 = nn.Identity()
                    basic_block.relu1 = nn.Identity()

                    for sub_block_name, sub_block in basic_block.named_children():
                        if "shortcut" in sub_block_name:
                            if len(sub_block) > 0:
                                cbr = list(sub_block)
                                cbr.append(nn.Identity())
                                fsc = self.conv_bn_relu(cbr)
                                
                                # update shortcut
                                setattr(basic_block, sub_block_name, fsc)
        return self.model