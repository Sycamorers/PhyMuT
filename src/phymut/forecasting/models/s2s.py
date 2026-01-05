import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class s2s(nn.Module):
    def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
        super().__init__()
        self.register_buffer("coeff", torch.tensor([0.3,0.5,0.9,1.0]).view(1,1,4))
        # self.register_buffer("coeff", torch.tensor([0.2,0.4,0.8,0.9]).view(1,1,4))
        self.encoder = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True, bidirectional=True)
        # attention: score = v^T tanh(W1 h_enc + W2 h_dec)
        self.W1 = nn.Linear(2*hid_dim, hid_dim)
        self.W2 = nn.Linear(hid_dim, hid_dim)
        self.v  = nn.Linear(hid_dim, 1, bias=False)
        self.decoder = nn.LSTMCell(in_dim, hid_dim)
        self.fc_out  = nn.Linear(hid_dim, forecast_len)

    def forward(self, x):
        x_scaled = x.clone(); x_scaled[:, :, :4] *= self.coeff
        # encode
        enc_out, (h_n, c_n) = self.encoder(x_scaled)  # enc_out: (b,seq,2*hid)
        # init decoder state with sum of bi-directional h
        h_dec = (h_n[0] + h_n[1])
        c_dec = (c_n[0] + c_n[1])
        # use last input as first decoder input
        inp = x_scaled[:, -1, :]
        # one-step decode with attention to produce forecast
        # compute attention scores
        score = self.v(torch.tanh(self.W1(enc_out) +
                                  self.W2(h_dec).unsqueeze(1)))  # (b,seq,1)
        alpha = torch.softmax(score, dim=1)               # (b,seq,1)
        context = (alpha * enc_out).sum(dim=1)            # (b,2*hid)
        # decoder step
        h_dec, c_dec = self.decoder(inp, (h_dec, c_dec))
        out = self.fc_out(h_dec + context[:, :h_dec.size(-1)])
        return out
    
# class s2s(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.encoder = nn.LSTM(in_dim, hid_dim, num_layers,
#                                batch_first=True, bidirectional=True)
#         self.W1 = nn.Linear(2*hid_dim, hid_dim)
#         self.W2 = nn.Linear(hid_dim, hid_dim)
#         self.v  = nn.Linear(hid_dim, 1, bias=False)
#         self.decoder = nn.LSTMCell(in_dim, hid_dim)
#         self.fc_out  = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)
#         enc_out, (h_n, c_n) = self.encoder(x_scaled)
#         h_dec = (h_n[0] + h_n[1])
#         c_dec = (c_n[0] + c_n[1])
#         inp = x_scaled[:, -1, :]
#         score = self.v(torch.tanh(self.W1(enc_out) +
#                                   self.W2(h_dec).unsqueeze(1)))
#         alpha = torch.softmax(score, dim=1)
#         context = (alpha * enc_out).sum(dim=1)
#         h_dec, c_dec = self.decoder(inp, (h_dec, c_dec))
#         return self.fc_out(h_dec + context[:, :h_dec.size(-1)])

# class s2s(nn.Module):
#     def __init__(self, in_dim, hid_dim, forecast_len, num_layers=1):
#         super().__init__()
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         self.encoder = nn.LSTM(in_dim, hid_dim, num_layers,
#                                batch_first=True, bidirectional=True)
#         self.W1 = nn.Linear(2*hid_dim, hid_dim)
#         self.W2 = nn.Linear(hid_dim, hid_dim)
#         self.v  = nn.Linear(hid_dim, 1, bias=False)
#         self.decoder = nn.LSTMCell(in_dim, hid_dim)
#         self.fc_out  = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x):
#         enc_out, (h_n, c_n) = self.encoder(x)
#         h_dec = (h_n[0] + h_n[1])
#         c_dec = (c_n[0] + c_n[1])
#         inp = x[:, -1, :]
#         score = self.v(torch.tanh(self.W1(enc_out) +
#                                   self.W2(h_dec).unsqueeze(1)))
#         alpha = torch.softmax(score, dim=1)
#         context = (alpha * enc_out).sum(dim=1)
#         h_dec, c_dec = self.decoder(inp, (h_dec, c_dec))
#         return self.fc_out(h_dec + context[:, :h_dec.size(-1)])