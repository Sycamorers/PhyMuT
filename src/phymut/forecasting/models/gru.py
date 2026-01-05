import torch
import torch.nn as nn

# gru_model.py
class gru(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
        super().__init__()
        self.register_buffer(
            # "coeff", torch.tensor([0.2, 0.4, 0.8, 0.9]).view(1,1,4)
            "coeff", torch.tensor([0.3, 0.5, 0.9, 1.0]).view(1, 1, 4)
            
        )
        self.gru = nn.GRU(in_dim, hid_dim, num_layers, batch_first=True)
        self.fc  = nn.Linear(hid_dim, forecast_len)

    def forward(self, x):
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff
        _, h = self.gru(x_scaled)
        last = h[-1]
        return self.fc(last)
  
    
# class gru(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         # learnable scaling for first 4 features
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.gru = nn.GRU(in_dim, hid_dim, num_layers, batch_first=True)
#         self.fc  = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         # x: (batch, seq_len, in_dim)
#         # apply learned scaling to first 4 features
#         scaled = self.feature_scale(x[:, :, :4])  # (batch, seq_len, 4)
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         _, h = self.gru(x_scaled)
#         last = h[-1]
#         return self.fc(last)
    
    
    
# class gru(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         # learnable scaling for first 4 features
#         self.gru = nn.GRU(in_dim, hid_dim, num_layers, batch_first=True)
#         self.fc  = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         # x: (batch, seq_len, in_dim)
#         _, h = self.gru(x)
#         last = h[-1]
#         return self.fc(last)