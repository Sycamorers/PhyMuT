import torch
import torch.nn as nn

# lstm_model.py
class lstm(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hid_dim, forecast_len)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        last = h[-1]
        return self.fc(last)
    
# class lstm(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
#         self.fc   = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         _, (h, _) = self.lstm(x_scaled)
#         last = h[-1]
#         return self.fc(last)
    
    
# class lstm(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
#         self.fc   = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         _, (h, _) = self.lstm(x)
#         last = h[-1]
#         return self.fc(last)