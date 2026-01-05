from .nn.modules import GroupAttention
import torch
import torch.nn as nn

class HybridMultiTaskModel(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
        super().__init__()
        self.in_dim  = in_dim
        self.hid_dim = hid_dim

        self.lstm1 = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        self.attn  = GroupAttention(hid_dim, in_dim)

        # ← postpone lstm2 construction until we know the true width
        self.lstm2 = None

        self.fc    = nn.Linear(hid_dim, forecast_len)

        self.register_buffer(
            "coeff", torch.tensor([0.3, 0.5, 0.9, 1.0]).view(1, 1, 4)
            # "coeff", torch.tensor([0.2, 0.4, 0.8, .9]).view(1, 1, 4)
        )

    def forward(self, x):
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff

        out1, (h1, _) = self.lstm1(x_scaled)
        last_hidden   = h1[-1].unsqueeze(1).expand_as(out1)
        attn_out      = self.attn(last_hidden, x_scaled)

        non_yield     = x_scaled[:, :, 4:]
        x_lstm2_in    = torch.cat([attn_out, non_yield], dim=-1)

        # Build lstm2 on first forward pass with the right width
        if self.lstm2 is None:
            in_size = x_lstm2_in.size(-1)
            self.lstm2 = nn.LSTM(
                input_size=in_size,
                hidden_size=self.hid_dim,
                num_layers=1,
                batch_first=True
            ).to(x.device)

        out2, _ = self.lstm2(x_lstm2_in)
        return self.fc(out2[:, -1, :])
    
    
# class HybridMultiTaskModel(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.lstm1 = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
#         self.attn  = GroupAttention(hid_dim, in_dim)
#         # in_dim - 4 additional features
#         self.lstm2 = nn.LSTM(hid_dim + (in_dim - 4), hid_dim,
#                              num_layers, batch_first=True)
#         self.fc    = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         out1, (h1, _) = self.lstm1(x_scaled)
#         last_hidden   = h1[-1].unsqueeze(1).expand_as(out1)
#         attn_out      = self.attn(last_hidden, x_scaled)
#         non_yield     = x_scaled[:, :, 4:]
#         x2            = torch.cat([attn_out, non_yield], dim=-1)
#         out2, _       = self.lstm2(x2)
#         return self.fc(out2[:, -1, :])
    
    
    
# class HybridMultiTaskModel(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.lstm1 = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
#         self.attn  = GroupAttention(hid_dim, in_dim)
#         # in_dim - 4 additional features
#         self.lstm2 = nn.LSTM(hid_dim + (in_dim - 4), hid_dim,
#                              num_layers, batch_first=True)
#         self.fc    = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         out1, (h1, _) = self.lstm1(x)
#         last_hidden   = h1[-1].unsqueeze(1).expand_as(out1)
#         attn_out      = self.attn(last_hidden, x)
#         non_yield     = x[:, :, 4:]
#         x2            = torch.cat([attn_out, non_yield], dim=-1)
#         out2, _       = self.lstm2(x2)
#         return self.fc(out2[:, -1, :])