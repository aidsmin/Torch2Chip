"""
Torch to chip
"""

import numpy as np 
import torch
import copy
import torch.nn as nn
from torch import Tensor
from methods import MulShift, CPUQBaseConv2d, QBaseConv2d, ConvBNReLU
from .fuser import LayerFuser
from fxpmath import Fxp

class T2C(object):
    """
    Deploying the pretrained Pytorch model to hardware-feasible parameters: 
    - Layer fusion
    - Integer conversion
    - Parameter saving
    - Define the precision of the high precision scaling / shifting

    Args:
    - model: Pretrained DNN model (after fusion)
    - swl: World length of the high precision scaling/shifting factor
    - swl: Fractional bits the high precision scaling/shifting factor
    """
    def __init__(self, model:nn.Module, swl:int, sfl:int, args):
        self.swl = swl
        self.sfl = sfl
        self.args = args

        # model fusion
        fuser = LayerFuser(model)
        fuser.layers()
        fused_model = fuser.fuse()
        fuser.inference(fused_model)

        # integer conversion
        qnn = self.scale_bias2int(fused_model)
        self.model = qnn

        print("\n======== T2C: Torch to chip ========")

    def f2fxp(self, val):
        vfix = Fxp(val, signed=True, n_word=self.swl, n_frac=self.sfl)
        vfix = vfix.base_repr(10)
        vnp = np.array(vfix).astype(float)
        return torch.from_numpy(vnp).cuda()

    def scale_bias2int(self, model:nn.Module):
        """
        Convert the pre-computed scaling factor and bias to high precision integer
        """
        qnn = copy.deepcopy(model)
        for n, m in qnn.named_modules():
            if isinstance(m, MulShift):
                m.fl = self.sfl
                scale = m.scale.cpu().numpy()
                bias = m.bias.cpu().numpy()

                # to numpy
                sint = self.f2fxp(scale)
                bint = self.f2fxp(bias)
                
                # insert back
                m.scale = sint.float()
                m.bias = bint.float()
        return qnn
    
    def get_info(self, model:nn.Module):
        nparams = 0.
        fm_size = []
        for n, v in model.state_dict().items():
            if 'qweight' in n:
                nparams += v.numel()
            elif "fm_max" in n:
                fm_size.append(int(v.item()))
        print("Number of weight parameters = {}".format(int(nparams)))
        print("Maximum feature map size = {} bit".format(max(fm_size)))

    def nn2chip(self):
        return self.model
    
    def nn2cpu(self, fused_nn:nn.Module):
        cpu_nn = copy.deepcopy(fused_nn)
        for n, m in cpu_nn.named_modules():
            if isinstance(m, ConvBNReLU):
                conv = m.conv
                qconv = CPUQBaseConv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                                        wbit=conv.wbit, abit=conv.abit, train_flag=conv.train_flag)
                setattr(qconv, "aq", conv.aq)
                setattr(qconv, "wq", conv.wq)
                
                # insert the integer weight
                qconv.qint8(conv.qweight)

                # update conv module
                setattr(m, "conv", qconv)
        return cpu_nn