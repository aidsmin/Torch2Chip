"""
Trainer for sparsification
"""

import torch.nn as nn
from .trainer import BaseTrainer
from pruner import Pruner, NMPruner

PRUNERS = {
    "mp": Pruner,
    "nm": NMPruner
}

class SparseTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, loss_type: str, trainloader, validloader, args, logger):
        super(SparseTrainer, self).__init__(model, loss_type, trainloader, validloader, args, logger)

        # pruner
        self.pruner = PRUNERS[str(self.args.pruner)](self.model, self.trainloader, args=self.args)

    def train_step(self, inputs, target):
        out, loss = super().train_step(inputs, target)
        self.pruner.step()
        return out, loss
    
    def train_epoch(self):
        super().train_epoch()
        self.logger_dict["sparsity"] = self.pruner.sparsity()
        self.logger_dict["pr"] = 1 - self.pruner.pr


