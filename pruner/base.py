"""
Base pruning method
"""

import torch
import torch.nn as nn

class Pruner(object):
    r"""Basic score-based pruner with magnitude-based element-wise pruning. 

    Args:
    model: Neural network model (type: nn.Module). 
    loader: Dataloader of training.
    args: Argparse argument. 
    interval: Sparsity update frequency (default = 1,000 iterations).
    
    Attributes:
    masks (Dict): Dictionary-based buffer stores layer-wise masks. 
    layer2sparse (Dict): Dictionary-based layer-wise sparsity. 
    steps (int): Steps tracker
    init_density (float): Starting density of the model. 
    final_density (float): Target density of the model.
    """
    def __init__(self, model:nn.Module, loader=None, args=None, interval=1000):
        self.model = model
        self.args = args
        self.loader = loader

        # masks
        self.masks = {}

        # stats
        self.layer2sparse = {}

        # steps
        self.steps = 0
        self.prune_every_k_steps = interval

        # initialization
        self.init_prune_epoch = self.args.init_prune_epoch
        self.ini_iter = int((self.init_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        self.final_iter = int((self.args.final_prune_epoch * len(self.loader)) / self.prune_every_k_steps)
        self.total_prune_iter = self.final_iter - self.ini_iter

        # sparsity/density 
        self.final_density = self.args.final_density
        self.init_density = self.args.init_density
        self.pr = self.args.init_density

        # register the initial mask
        self.reg_masks()

    def sparsity(self):
        r"""Return the overall element-wise sparsity. 
        """
        total, nz = 0, 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                total += m.mask.numel()
                nz += m.mask.sum()
        return 1 - nz / total

    def prune_rate_step(self):
        r"""Update the prune rate.
        """
        self.curr_prune_iter = int(self.steps / self.prune_every_k_steps)
        ramping_decay = (1 - ((self.curr_prune_iter - self.ini_iter) / self.total_prune_iter)) ** 3
        curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.final_density) * (1 - ramping_decay)
        return curr_prune_rate

    def reg_masks(self):
        r"""Fetch the layer-wise mask from each sparse convolutional layer.
        """
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.data
                mask = torch.ones_like(weight)
                self.masks[n] = mask
    
    def _layer_stats(self):
        r"""Fetch the layer-wise sparsity.
        """
        self.name2density = {}

        for name, mask in self.masks.items():
            self.name2nonzeros[name] = mask.sum().item() / mask.numel()

    def collect_score(self):
        r"""Collect the layer-wise magnitude score. 
        """
        weight_abs = []
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight_abs.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        return mp_scores
    
    def apply_masks(self):
        r"""Assign the updated mask to each sparse convolutional layer. 
        """
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                mask = self.masks[n]
                m.mask = mask
    
    def update_mask(self, threshold):
        r"""Update the sparse mask.
        """
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                self.masks[n] = m.weight.abs().gt(threshold).float()
    
    def prune(self):
        r"""Prune step: Compute the threshold and update the masks. 
        """
        # pruning schedule
        self.pr = self.prune_rate_step()
        
        if self.curr_prune_iter >= self.ini_iter and self.curr_prune_iter <= self.final_iter:

            # collect score (element score / group score)
            scores = self.collect_score()
            nkeep = int(len(scores) * (1 - self.pr))
            
            # sort the scores
            if nkeep != 0:
                topkscore, _ = torch.topk(scores, nkeep, sorted=True)                
                threshold = topkscore[-1]
            else:
                threshold = scores.max().item() + 1e-3

            # update and apply masks
            self.update_mask(threshold)
        else:
            self.pr = 0.0

    def step(self):
        r"""Take the sparsification step. 
        """
        self.steps += 1
        if self.steps >= len(self.loader) and self.steps % self.prune_every_k_steps == 0:
            self.prune()
            self.apply_masks()