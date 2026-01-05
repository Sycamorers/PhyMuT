import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, backcast_len):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.theta_b = nn.Linear(hid_dim, backcast_len)
        self.theta_f = nn.Linear(hid_dim, forecast_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, in_dim)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        backcast = self.theta_b(h)
        forecast = self.theta_f(h)
        return backcast, forecast
    

class nbeats(nn.Module):
    def __init__(self, in_dim, seq_len, hid_dim, forecast_len, backcast_len, n_blocks=3):
        super().__init__()
        self.seq_len = seq_len
        self.in_dim = in_dim
        self.blocks = nn.ModuleList([
            NBeatsBlock(seq_len * in_dim, hid_dim, forecast_len, backcast_len)
            for _ in range(n_blocks)
        ])
        self.register_buffer(
            "coeff",
            torch.tensor([0.3, 0.5, 0.9, 1.0], dtype=torch.float32).view(1, 1, 4)
            # torch.tensor([0.2, 0.4, 0.8, 0.9], dtype=torch.float32).view(1, 1, 4)
        )

    def forward(self, x):
        # flatten: concatenate all timesteps and dims
        # Note: x is (b,seq,in), so flatten to (b, seq*in)
        b, seq, d = x.size()
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff
        resid = x_scaled.reshape(b, -1)
        total_f = 0
        for block in self.blocks:
            bcast, fcast = block(resid)
            resid = resid - bcast
            total_f = total_f + fcast
        return total_f
    
    
    
# class NBeatsBlock(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, backcast_len):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hid_dim)
#         self.fc2 = nn.Linear(hid_dim, hid_dim)
#         self.theta_b = nn.Linear(hid_dim, backcast_len)
#         self.theta_f = nn.Linear(hid_dim, forecast_len)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         h = self.relu(self.fc1(x))
#         h = self.relu(self.fc2(h))
#         backcast = self.theta_b(h)
#         forecast = self.theta_f(h)
#         return backcast, forecast

    
# class nbeats(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, backcast_len, seq_len, n_blocks=3):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         # n_blocks of NBeatsBlock expect flattened input
#         self.blocks = nn.ModuleList([
#             NBeatsBlock(in_dim * seq_len, hid_dim, forecast_len, backcast_len)
#             for _ in range(n_blocks)
#         ])

#     def forward(self, x):
#         b, seq, d = x.size()
#         # scale first 4 features across all timesteps
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         resid = x_scaled.reshape(b, -1)
#         total_f = 0
#         for block in self.blocks:
#             bcast, fcast = block(resid)
#             resid = resid - bcast
#             total_f = total_f + fcast
#         return total_f
    
    
    

# class NBeatsBlock(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, backcast_len):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hid_dim)
#         self.fc2 = nn.Linear(hid_dim, hid_dim)
#         self.theta_b = nn.Linear(hid_dim, backcast_len)
#         self.theta_f = nn.Linear(hid_dim, forecast_len)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         h = self.relu(self.fc1(x))
#         h = self.relu(self.fc2(h))
#         backcast = self.theta_b(h)
#         forecast = self.theta_f(h)
#         return backcast, forecast

    
# class nbeats(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, backcast_len, seq_len, n_blocks=3):
#         super().__init__()
#         self.blocks = nn.ModuleList([
#             NBeatsBlock(in_dim * seq_len, hid_dim, forecast_len, backcast_len)
#             for _ in range(n_blocks)
#         ])

#     def forward(self, x):
#         b, seq, d = x.size()
#         # scale first 4 features across all timesteps
#         resid = x.reshape(b, -1)
#         total_f = 0
#         for block in self.blocks:
#             bcast, fcast = block(resid)
#             resid = resid - bcast
#             total_f = total_f + fcast
#         return total_f