"""
Torch to chip
"""

import numpy as np 
import torch
import copy
import torch.nn as nn
from torch import Tensor
from methods import MulShift
from fxpmath import Fxp


class T2C(object):
    """
    Deploying the pretrained Pytorch model to hardware-feasible parameters: 
    - Integer conversion
    - Parameter saving
    - Define the precision of the high precision scaling / shifting

    Args:
    - model: Pretrained DNN model (after fusion)
    - swl: World length of the high precision scaling/shifting factor
    - swl: Fractional bits the high precision scaling/shifting factor
    """
    def __init__(self, model:nn.Module, swl:int, sfl:int, save:bool=False):
        self.model = model
        self.swl = swl
        self.sfl = sfl

    def f2fxp(self, val):
        vfix = Fxp(val, signed=True, n_word=self.swl, n_frac=self.sfl)
        vfix = vfix.base_repr(10)
        vnp = np.array(vfix).astype(float)
        return torch.from_numpy(vnp).cuda()

    def scale_bias2int(self):
        """
        Convert the pre-computed scaling factor and bias to high precision integer
        """
        qnn = copy.deepcopy(self.model)
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


    def save_weights(self):
        """
        Save the quantized integer weights (from the registered buffer) to external files (e.g., .pt, .npy)
        """
        pass

    def save_scale_bias(self):
        """
        Save the high precision scaling factor and bias to external files
        """
        pass

    def get_buffer_size(self):
        """
        Find the required maximum precision of the intermediate results
        """
        pass


