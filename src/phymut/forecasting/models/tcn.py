import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

    
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch,
                                           kernel_size,
                                           padding=padding,
                                           dilation=dilation))
        self.relu1 = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch,
                                           kernel_size,
                                           padding=padding,
                                           dilation=dilation))
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        out = self.relu1(self.conv1(x))[:, :, :-self.conv1.padding[0]]
        out = self.relu2(self.conv2(out))[:, :, :-self.conv2.padding[0]]
        res = self.downsample(x) if self.downsample else x
        return out + res

class tcn(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_levels=4,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.register_buffer("coeff", torch.tensor([0.3,0.5,0.9,1.0]).view(1,1,4))
        # self.register_buffer("coeff", torch.tensor([0.2,0.4,0.8,0.9]).view(1,1,4))
        # project features -> hid_dim channels
        self.input_proj = nn.Conv1d(in_dim, hid_dim, kernel_size=1)
        # temporal blocks with exponentially increasing dilation
        self.tcn = nn.Sequential(*[
            nn.Sequential(
                TCNBlock(hid_dim, hid_dim, kernel_size, dilation=2**i),
                nn.Dropout(dropout)
            ) for i in range(num_levels)
        ])
        self.fc = nn.Linear(hid_dim, forecast_len)

    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff
        # to (batch, channels, seq_len)
        h = self.input_proj(x_scaled.transpose(1,2))
        h = self.tcn(h)
        # take last time step
        last = h[:, :, -1]      # (batch, hid_dim)
        return self.fc(last)


# class TCNBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size, dilation):
#         super().__init__()
#         padding = (kernel_size - 1) * dilation
#         self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
#                                            padding=padding, dilation=dilation))
#         self.relu1 = nn.ReLU()
#         self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
#                                            padding=padding, dilation=dilation))
#         self.relu2 = nn.ReLU()
#         self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
#         self.init_weights()

#     def init_weights(self):
#         for m in [self.conv1, self.conv2]:
#             nn.init.kaiming_normal_(m.weight)

#     def forward(self, x):
#         # x: (batch, channels, seq_len)
#         out = self.relu1(self.conv1(x))[:, :, :-self.conv1.padding[0]]
#         out = self.relu2(self.conv2(out))[:, :, :-self.conv2.padding[0]]
#         res = self.downsample(x) if self.downsample else x
#         return out + res
    
# class tcn(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_levels=4, kernel_size=3, dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Conv1d(in_dim, hid_dim, kernel_size=1)
#         self.tcn = nn.Sequential(*[
#             nn.Sequential(
#                 TCNBlock(hid_dim, hid_dim, kernel_size, dilation=2**i),
#                 nn.Dropout(dropout)
#             ) for i in range(num_levels)
#         ])
#         self.fc = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         h = self.input_proj(x.transpose(1, 2))
#         h = self.tcn(h)
#         last = h[:, :, -1]
#         return self.fc(last)