"""
Base pruning method
"""

import torch
import torch.nn as nn
from typing import List

class Pruner(object):
    def __init__(self, model:nn.Module, args=None):
        self.model = model
        