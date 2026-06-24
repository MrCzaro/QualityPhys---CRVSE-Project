"""
CRVSE PhysFormer architecture for rPPG heart rate esimation.

This file rebuilds the best model architecture outside the traning notebook,

Expected current model input:
    x.shape = (batch, 3, 240)
    
    
Channels:
    0 = POS
    1 = CHROM
    2 = GREEN
    
Output:
    HR estimate in BPM, shape = (batch,)
"""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer time tokens.

    Limitation:
        This encoding assumes a maximum supported token length. Current checkpoint
        uses max_len=300 and target_frames=240.
    """

    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding.

        Parameters
        ----------
        x:
            Tensor with shape (batch, time, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with shape (batch, time, d_model).
        """

        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {x.size(1)} exceeds positional encoding "
                f"max length {self.pe.size(1)}."
            )

        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayerCustom(nn.Module):
    """
    Custom pre-norm transformer encoder layer.

    This mirrors the training notebook version used by the checkpoint.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = 256, 
                 dropout: float = 0.1) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            batch_size,
            time_steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            time_steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            time_steps,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, time_steps, -1)

        return self.out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one transformer encoder layer.

        Parameters
        ----------
        x:
            Tensor with shape (batch, time, d_model).

        Returns
        -------
        torch.Tensor
            Tensor with shape (batch, time, d_model).
        """

        x = x + self.dropout(self._attention(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))

        return x


class CRVSEPhysFormer(nn.Module):
    """
    CNN + FFT + Transformer model for rPPG heart-rate estimation.

    Limitation:
        The FFT projection size depends on target_frames. The current checkpoint was
        trained with target_frames=240, so strict checkpoint loading requires the same
        value.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_channels: int = 16,
        freq_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.11331939348791525,
        hr_min: float = 40.0,
        hr_max: float = 180.0,
        target_frames: int = 240,
        max_positional_length: int = 300,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.cnn_channels = cnn_channels
        self.freq_channels = freq_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.target_frames = target_frames

        self.d_model = cnn_channels + freq_channels

        if self.d_model % n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={n_heads}.")

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                cnn_channels // 2,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(cnn_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                cnn_channels // 2,
                cnn_channels,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        n_fft_bins = target_frames // 2 + 1

        self.freq_proj = nn.Sequential(
            nn.Linear(n_fft_bins, freq_channels * 4),
            nn.ReLU(),
            nn.Linear(freq_channels * 4, freq_channels),
        )

        self.pos_enc = PositionalEncoding(
            d_model=self.d_model,
            max_len=max_positional_length,
            dropout=dropout,
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayerCustom(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate HR from multichannel rPPG windows.

        Parameters
        ----------
        x:
            Tensor with shape (batch, channels, target_frames).

        Returns
        -------
        torch.Tensor
            Estimated HR in BPM with shape (batch,).
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape (batch, channels, time), got {tuple(x.shape)}."
            )

        batch_size, channels, time_steps = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}."
            )

        if time_steps != self.target_frames:
            raise ValueError(
                f"Expected target_frames={self.target_frames}, got {time_steps}."
            )

        time_feat = self.encoder(x)
        time_feat = time_feat.permute(0, 2, 1)

        freq = torch.fft.rfft(x, norm="ortho")
        freq_mag = freq.abs().mean(dim=1)
        freq_feat = self.freq_proj(freq_mag)
        freq_feat = freq_feat.unsqueeze(1).expand(-1, time_steps, -1)

        combined = torch.cat([time_feat, freq_feat], dim=-1)
        combined = self.pos_enc(combined)

        for layer in self.transformer_layers:
            combined = layer(combined)

        out = combined.mean(dim=1)
        out = self.head(out).squeeze(-1)

        if not self.training:
            out = out.clamp(self.hr_min, self.hr_max)

        return out


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)