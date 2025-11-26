import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Expert(nn.Module):
    """Single expert network (essentially a FeedForward layer)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))