"""
Pruner with N:M Structured Fine-grained Sparsity

Accelerating Sparse Deep Neural Networks: https://arxiv.org/pdf/2104.08378.pdf
"""

import torch
import torch.nn as nn
from .base import Pruner
from torch import Tensor

class NMPruner(Pruner):
    r"""Strutured fine-grained sparsification pruner.

    Args:
    model: Neural network model (type: nn.Module). 
    loader: Dataloader of training.
    args: Argparse argument. 
    interval: Sparsity update frequency (default = 1,000 iterations).

    Attributes:
    M (int): Size of sparse groups (divisible by 4).
    N (int): Number of sparse elements inside.
    nchw: Flag of exploiting sparsity along the input channel.
    """
    def __init__(self, model: nn.Module, loader=None, args=None, interval=1000):
        super().__init__(model, loader, args, interval)
        self.N = args.N
        self.M = args.M

        assert self.M > self.N, "# of Sparse elements (N) cannot be greater or equal to the group size (M)."

        # input channel our output channel oriented
        self.nchw = self.args.nchw

    def get_groups(self, tensor:Tensor):
        r"""Compute the total number of groups of the given weight tensor
        """
        length = tensor.numel()
        groups = int(length / self.M)
        return groups

    def reshape2d(self, weight:Tensor, group:int):
        r"""Reshape the 4-D weight tensor into the size of [# of groups, M].
        """
        # groups
        if not self.nchw:
            wtemp = weight.detach().abs().permute(1,2,3,0).reshape(group, self.M)   # output channel oriented
        else:
            wtemp = weight.detach().abs().permute(0,2,3,1).reshape(group, self.M)   # input channel oriented
        return wtemp

    def reshape4d(self, mask2d:Tensor, weight:Tensor):
        r"""Reshape the 2-D masks back to the 4-D shape as the weights. 
        """
        if self.nchw:
            mask4d = mask2d.reshape(weight.permute(0,2,3,1).shape)
            mask4d = mask4d.permute(0,3,1,2)
        else:
            mask4d = mask2d.reshape(weight.permute(1,2,3,0).shape)
            mask4d = mask4d.permute(3,0,1,2)
        return mask4d
    
    def collect_score(self):
        r"""Collect the group scores globally across the entire network. 
        """
        gscores = []
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.clone()
                group = self.get_groups(weight)

                wtemp = self.reshape2d(weight, group)                
                gscores.append(wtemp.sum(dim=1))
        
        mg_scores = torch.cat([torch.flatten(x) for x in gscores])
        return mg_scores

    def mn(self, weight:Tensor):
        r"""Sparsify N elements inside each M-sized group. 
        """
        index = torch.argsort(weight, dim=1)[:, :int(self.M-self.N)]
        w_b = torch.ones(weight.shape, device=weight.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0)
        return w_b
    
    def update_mask(self, threshold):
        r"""Update sparse masks of each layer. 
        """
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.clone()
                groups = self.get_groups(weight)
                wg = self.reshape2d(weight, groups)

                # 2-D mask
                mask2d = self.mn(wg)

                # make the important group dense
                if 1 - self.pr < 1.0:
                    dense = wg.sum(dim=1).gt(threshold)
                    mask2d[dense] = 1.0
                
                mask4d = self.reshape4d(mask2d, weight)
                self.masks[n] = mask4d
