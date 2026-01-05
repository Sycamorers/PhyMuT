import torch
import torch.nn as nn

# transformer_model.py
class transformer(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1, nhead=4):
        super().__init__()
        self.register_buffer(
            "coeff", torch.tensor([0.3, 0.5, 0.9, 1.0]).view(1,1,4)
            # "coeff", torch.tensor([0.2, 0.4, 0.8, 0.9]).view(1,1,4)
        )
        self.input_proj  = nn.Linear(in_dim, hid_dim)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc          = nn.Linear(hid_dim, forecast_len)

    def forward(self, x):
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff
        emb = self.input_proj(x_scaled)
        out = self.transformer(emb)
        last = out[:, -1, :]
        return self.fc(last)
    
    
    
# class transformer(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1, nhead=4):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.input_proj  = nn.Linear(in_dim, hid_dim)
#         encoder_layer    = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc          = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         emb = self.input_proj(x_scaled)
#         out = self.transformer(emb)
#         last = out[:, -1, :]
#         return self.fc(last)


# class transformer(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1, nhead=4):
#         super().__init__()
#         self.input_proj  = nn.Linear(in_dim, hid_dim)
#         encoder_layer    = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.fc          = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         emb = self.input_proj(x)
#         out = self.transformer(emb)
#         last = out[:, -1, :]
#         return self.fc(last)
