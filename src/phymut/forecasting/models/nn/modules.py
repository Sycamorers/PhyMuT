import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class GroupAttention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()

        # decide a split automatically
        split1 = min(120, input_dim)              # <= input_dim
        split2 = input_dim - split1               # remainder

        self.split1 = split1                      # remember for forward
        self.split2 = split2

        self.linear_group1 = nn.Linear(hidden_dim + split1, split1)
        self.mlp_group2   = nn.Sequential(
            nn.Linear(hidden_dim + split2, 64),
            nn.ReLU(),
            nn.Linear(64, split2)
        )

    def forward(self, hidden_state, x):
        yield_input = x                           # (B,T,input_dim)

        group1 = yield_input[:, :, :self.split1]  # (B,T,split1)
        group2 = yield_input[:, :, self.split1:]  # (B,T,split2)

        # group 1
        comb1  = torch.cat([hidden_state, group1], dim=-1)
        w1     = torch.softmax(self.linear_group1(comb1), dim=-1)
        att1   = w1 * group1

        # group 2
        comb2  = torch.cat([hidden_state, group2], dim=-1)
        w2     = torch.softmax(self.mlp_group2(comb2), dim=-1)
        att2   = w2 * group2

        return torch.cat([att1, att2], dim=-1)    # (B,T,input_dim)
