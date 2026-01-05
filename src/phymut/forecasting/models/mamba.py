import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

class mamba(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        forecast_len: int,
        num_layers: int = 2,
        **kwargs
    ):
        """
        A time-series forecaster that runs your input through a stacked Mamba,
        then takes the final time-step and projects to `forecast_len`.
        """
        super().__init__()

        # same coefficient scaling on the first 4 features
        self.register_buffer(
            "coeff",
            torch.tensor([0.3, 0.5, 0.9, 1.0], dtype=torch.float32).view(1, 1, 4)
            # torch.tensor([0.2, 0.4, 0.8, 0.9], dtype=torch.float32).view(1, 1, 4)
        )

        # Project input features to model dimension
        self.input_proj = nn.Linear(in_dim, hid_dim)

        # Build a Mamba stack: use hid_dim as the model width
        config = MambaConfig(
            d_model=hid_dim,
            n_layers=num_layers,
            **{k: v for k, v in kwargs.items() if k in MambaConfig.__annotations__}
        )
        self.mamba = Mamba(config)

        # Single output head (like other models)
        self.fc = nn.Linear(hid_dim, forecast_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        returns: (B, forecast_len)
        """
        # 1) scale the first 4 input channels
        x_scaled = x.clone()
        x_scaled[:, :, :4] *= self.coeff

        # 2) project to model dimension
        x_proj = self.input_proj(x_scaled)  # (B, T, hid_dim)

        # 3) run through Mamba layers → (B, T, hid_dim)
        y = self.mamba(x_proj)

        # 4) take last time-step and project
        last = y[:, -1, :]            # (B, hid_dim)
        return self.fc(last)



# class mamba(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hid_dim: int,
#         forecast_len: int,
#         num_layers: int = 2,
#         **kwargs
#     ):
#         """
#         A time-series forecaster that runs your input through a stacked Mamba,
#         then takes the final time-step and projects to `forecast_len`.
#         """
#         super().__init__()

#         # same coefficient scaling on the first 4 features
#         self.feature_scale = nn.Linear(4, 4, bias=False)
#         # Project input features to model dimension
#         self.input_proj = nn.Linear(in_dim, hid_dim)

#         # Build a Mamba stack: use hid_dim as the model width
#         config = MambaConfig(
#             d_model=hid_dim,
#             n_layers=num_layers,
#             **{k: v for k, v in kwargs.items() if k in MambaConfig.__annotations__}
#         )
#         self.mamba = Mamba(config)

#         # Single output head (like other models)
#         self.fc = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, T, in_dim)
#         returns: (B, forecast_len)
#         """
#         # 1) scale the first 4 input channels
#         scaled = self.feature_scale(x[:, :, :4])
#         x_scaled = torch.cat([scaled, x[:, :, 4:]], dim=-1)

#         # 2) project to model dimension
#         x_proj = self.input_proj(x_scaled)  # (B, T, hid_dim)

#         # 3) run through Mamba layers → (B, T, hid_dim)
#         y = self.mamba(x_proj)

#         # 4) take last time-step and project
#         last = y[:, -1, :]            # (B, hid_dim)
#         return self.fc(last)
    
    

# class mamba(nn.Module):
#     def __init__(
#         self,
#         in_dim: int,
#         hid_dim: int,
#         forecast_len: int,
#         num_layers: int = 2,
#         **kwargs
#     ):
#         """
#         A time-series forecaster that runs your input through a stacked Mamba,
#         then takes the final time-step and projects to `forecast_len`.
#         """
#         super().__init__()

#         # Project input features to model dimension
#         self.input_proj = nn.Linear(in_dim, hid_dim)

#         # Build a Mamba stack: use hid_dim as the model width
#         config = MambaConfig(
#             d_model=hid_dim,
#             n_layers=num_layers,
#             **{k: v for k, v in kwargs.items() if k in MambaConfig.__annotations__}
#         )
#         self.mamba = Mamba(config)

#         # Single output head (like other models)
#         self.fc = nn.Linear(hid_dim, forecast_len)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, T, in_dim)
#         returns: (B, forecast_len)
#         """

#         # 2) project to model dimension
#         x_proj = self.input_proj(x)  # (B, T, hid_dim)

#         # 3) run through Mamba layers → (B, T, hid_dim)
#         y = self.mamba(x_proj)

#         # 4) take last time-step and project
#         last = y[:, -1, :]            # (B, hid_dim)
#         return self.fc(last)