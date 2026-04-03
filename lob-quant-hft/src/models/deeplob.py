"""
DeepLOB: Deep Learning for Limit Order Books
=============================================
Replication of Zhang et al. (2019) "DeepLOB: Deep Learning for Limit Order Books"
  - CNN feature extractor with inception-style modules
  - Bidirectional LSTM sequence model
  - 3-class output: {0=down, 1=stationary, 2=up}

Reference: https://arxiv.org/abs/1808.03668
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inception module (multi-scale convolution)
# ---------------------------------------------------------------------------

class InceptionModule(nn.Module):
    """
    Inception-style block with 1×1, 3×1, and 5×1 convolutions + pool branch.
    All branches are concatenated along the channel axis.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_c = out_channels // 4

        # Branch 1×1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_c, kernel_size=1),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Branch 3×1
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_c, kernel_size=1),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(branch_c, branch_c, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Branch 5×1
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_c, kernel_size=1),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(branch_c, branch_c, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
        )
        # Pool branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, branch_c, kernel_size=1),
            nn.BatchNorm2d(branch_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ---------------------------------------------------------------------------
# DeepLOB
# ---------------------------------------------------------------------------

class DeepLOB(nn.Module):
    """
    DeepLOB model.

    Input shape : (batch, window_size, n_features)
                  e.g. (B, 100, 40) for 10-level LOB

    Architecture
    ------------
    1. Reshape to 4D: (B, 1, T, F)
    2. Three convolutional blocks (each: Conv2d → BN → LeakyReLU → MaxPool)
    3. Inception module
    4. Bidirectional LSTM
    5. Linear classifier

    Parameters
    ----------
    n_features    : number of input features per tick (default 40 = 4×10 levels)
    num_classes   : prediction classes (default 3)
    conv_filters  : list of out_channels for the 3 conv blocks
    inception_out : output channels of inception module (must be divisible by 4)
    lstm_hidden   : LSTM hidden size
    lstm_layers   : number of LSTM layers
    dropout       : dropout probability (applied in LSTM and before classifier)
    """

    def __init__(
        self,
        n_features: int = 40,
        num_classes: int = 3,
        conv_filters: list[int] | None = None,
        inception_out: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if conv_filters is None:
            conv_filters = [32, 32, 32]

        # ── CNN blocks ──────────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_filters[0], kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(conv_filters[0]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[0], conv_filters[0], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[0]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[0], conv_filters[0], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[0]),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(conv_filters[1]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[1], conv_filters[1], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[1]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[1], conv_filters[1], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[1]),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=(1, 10)),
            nn.BatchNorm2d(conv_filters[2]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[2], conv_filters[2], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[2]),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(conv_filters[2], conv_filters[2], kernel_size=(4, 1)),
            nn.BatchNorm2d(conv_filters[2]),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ── Inception ───────────────────────────────────────────────────────
        self.inception = InceptionModule(conv_filters[2], inception_out)

        # ── LSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=inception_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # ── Classifier ──────────────────────────────────────────────────────
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)  # *2 for bidir

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)  — batch of LOB sequences

        Returns
        -------
        logits : (B, num_classes)
        """
        B, T, F = x.shape

        # → (B, 1, T, F)
        x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.conv1(x)   # (B, C1, T', F')
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception
        x = self.inception(x)  # (B, inception_out, T'', 1)

        # Remove spatial F-dim, permute for LSTM: (B, T'', inception_out)
        x = x.squeeze(-1).permute(0, 2, 1)

        # LSTM
        x, _ = self.lstm(x)             # (B, T'', 2*lstm_hidden)
        x = self.dropout(x[:, -1, :])  # take last time step

        return self.classifier(x)       # (B, num_classes)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        return F.softmax(self(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
