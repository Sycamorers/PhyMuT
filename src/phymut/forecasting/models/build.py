# models/build.py

from typing import Any

from .hybrid import HybridMultiTaskModel
from .rnn import rnn
from .lstm import lstm
from .transformer import transformer
from .mamba import mamba
from .gru import gru
from .s2s import s2s
from .tcn import tcn
from .nbeats import nbeats

def build_model(
    model_name: str,
    in_dim: int,
    hid_dim: int,
    forecast_len: int,
    num_layers: int = 2,
    **kwargs
):
    """
    Universal factory for your forecasting models.
    
    Args:
      model_name   : one of "hybrid", "rnn_direct", "lstm_direct", "transformer", "mamba", "gru"
      in_dim       : feature dimension per time step
      hid_dim      : hidden/state dimension
      forecast_len : number of steps to forecast
      num_layers   : for stacked RNN/LSTM/GRU layers (or hybrid attention depth)
      **kwargs     : model‐specific overrides (dropout, attention heads, etc.)
    """
    if model_name == "hybrid":
        return HybridMultiTaskModel(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "rnn":
        return rnn(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "lstm":
        return lstm(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "transformer":
        return transformer(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "mamba":
        return mamba(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "gru":
        return gru(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "s2s":
        return s2s(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_layers=num_layers,
        )
    elif model_name == "tcn":
        return tcn(
            in_dim=in_dim,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            num_levels=4,  # default value, can be adjusted
            kernel_size=3,  # default value, can be adjusted
            dropout=0.1,  # default value, can be adjusted
        )
    elif model_name == "nbeats":
        in_dim_kw = kwargs.get("in_dim", in_dim)
        seq_len = kwargs.get("seq_len", 1)
        backcast_len = seq_len * in_dim_kw
        return nbeats(
            in_dim=in_dim_kw,
            seq_len=seq_len,
            hid_dim=hid_dim,
            forecast_len=forecast_len,
            backcast_len=backcast_len,
            n_blocks=3,
        )
    else:
        
        
        raise ValueError(f"Unknown model_name: {model_name}")
