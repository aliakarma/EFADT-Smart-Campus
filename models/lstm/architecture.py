"""
EFADT — OccupancyLSTM Architecture
=====================================
2-layer LSTM for occupancy forecasting at the edge node.
  - Input:  lookback_steps × n_features (12 × 14)
  - Output: 1 (next-step occupancy forecast)

Designed to run locally on each building's edge node.
Only gradient updates leave the device during FL training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class OccupancyLSTM(nn.Module):
    """
    Two-layer LSTM occupancy forecasting model.

    Parameters
    ----------
    input_dim : int
        Number of input features (default: 14).
    hidden_size : int
        LSTM hidden state dimension (default: 128).
    num_layers : int
        Number of stacked LSTM layers (default: 2).
    dropout : float
        Dropout between LSTM layers (default: 0.2).
    output_dim : int
        Number of output values (default: 1 — occupancy count).
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for LSTM weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, input_dim)
        hidden : tuple of (h_0, c_0), optional
            Initial hidden and cell states.

        Returns
        -------
        (predictions, (h_n, c_n))
            predictions : shape (batch, output_dim)
        """
        out, hidden = self.lstm(x, hidden)
        # Use last timestep output
        out = self.dropout(out[:, -1, :])      # (batch, hidden_size)
        predictions = self.fc(out)             # (batch, output_dim)
        return predictions, hidden

    def predict(
        self,
        x: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convenience method for inference (no grad, clamp to [0, max_occ]).

        Parameters
        ----------
        x : torch.Tensor or np.ndarray, shape (batch, seq_len, input_dim)
        device : torch.device, optional

        Returns
        -------
        torch.Tensor, shape (batch,)
            Non-negative occupancy forecasts.
        """
        if device is None:
            device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)

        self.eval()
        with torch.no_grad():
            preds, _ = self(x)
        return torch.clamp(preds.squeeze(-1), min=0.0)

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def gradient_update(self) -> dict[str, torch.Tensor]:
        """
        Return current gradient update Δθ = {name: grad} for FL transmission.
        Called after local training to extract gradient for DP mechanism.
        """
        return {
            name: param.grad.clone().detach()
            for name, param in self.named_parameters()
            if param.grad is not None
        }


class CampusDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for sliding-window LSTM training.

    Parameters
    ----------
    features : np.ndarray, shape (n_timesteps, n_features)
        Normalized feature matrix.
    targets : np.ndarray, shape (n_timesteps,)
        Occupancy targets (next-step).
    lookback : int
        Sequence length (number of 30s steps to use as context).
    """

    def __init__(
        self,
        features,
        targets,
        lookback: int = 12,
    ) -> None:
        import numpy as np
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.X) - self.lookback

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.lookback]
        y_val = self.y[idx + self.lookback]
        return x_seq, y_val


def build_model(cfg: dict, device: Optional[torch.device] = None) -> OccupancyLSTM:
    """
    Factory function to build OccupancyLSTM from config dict.

    Parameters
    ----------
    cfg : dict
        Should contain keys: input_dim, hidden_size, num_layers, dropout, output_dim.
    device : torch.device, optional

    Returns
    -------
    OccupancyLSTM on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_cfg = cfg.get("lstm", cfg)
    model = OccupancyLSTM(
        input_dim=lstm_cfg.get("input_dim", 14),
        hidden_size=lstm_cfg.get("hidden_size", 128),
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.2),
        output_dim=lstm_cfg.get("output_dim", 1),
    )
    return model.to(device)


if __name__ == "__main__":
    import torch

    model = OccupancyLSTM()
    print(f"Model parameters: {model.get_parameter_count():,}")

    # Test forward pass
    x = torch.randn(8, 12, 14)   # batch=8, seq=12, features=14
    preds, (h, c) = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {preds.shape}")   # (8, 1)
    print(f"Hidden shape: {h.shape}")       # (2, 8, 128)
