"""
Pytorch JIT utils
"""

import os
import torch

def save_jit_model(model, save_path, filename):
    filepath = os.path.join(save_path, filename)
    torch.jit.save(torch.jit.script(model), filepath)
