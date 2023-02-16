"""
Base pruning method
"""

import torch
import torch.nn as nn
from torch import Tensor

class Pruner(object):
    """
    Basic element-wise pruning based on element-wise score
    """
    def __init__(self, model:nn.Module, args=None, interval=1000):
        self.model = model
        self.args = args

        # masks
        self.masks = {}

        # stats
        self.layer2sparse = {}

        # steps
        self.steps = 0

        # initialization
        self.init_prune_epoch = self.args.init_prune_epoch
        self.prune_every_k_steps = interval
        self.reg_masks()

    def prune_rate_decay(self):
        ini_iter = int((self.init_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        ramping_decay = (1 - ((self.curr_prune_iter - ini_iter) / self.total_prune_iter)) ** 3
        curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.final_density) * (1 - ramping_decay)
        return curr_prune_rate

    def reg_masks(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.data
                mask = torch.ones_like(weight)
                self.masks[n] = mask
    
    def _layer_stats(self):
        self.name2density = {}

        for name, mask in self.masks.items():
            self.name2nonzeros[name] = mask.sum().item() / mask.numel()

    def collect_score(self):
        weight_abs = []
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight_abs.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        return mp_scores
    
    def apply_masks(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                mask = self.masks[n]
                m.mask = mask
    
    def prune(self):
        pass