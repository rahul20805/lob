"""
LOB Transformer
================
Transformer encoder for LOB mid-price direction prediction.

Architecture
------------
  1. Linear projection of raw features → d_model
  2. Positional encoding (learnable or sinusoidal)
  3. N × TransformerEncoderLayer
  4. Global average pooling over sequence
  5. Linear classifier

Reference: "Attention Is All You Need" (Vaswani et al., 2017)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])

        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LearnablePE(nn.Module):
    """Learnable positional embeddings."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return self.dropout(x + self.pe(pos))


# ---------------------------------------------------------------------------
# Transformer encoder for LOB
# ---------------------------------------------------------------------------

class LOBTransformer(nn.Module):
    """
    Transformer-based LOB classifier.

    Input  : (B, T, n_features)
    Output : (B, num_classes)

    Parameters
    ----------
    n_features         : number of raw LOB features per tick
    d_model            : transformer hidden dimension
    nhead              : number of attention heads
    num_encoder_layers : number of TransformerEncoderLayer blocks
    dim_feedforward    : FFN inner dimension
    dropout            : dropout probability
    max_seq_len        : maximum sequence length (for positional encoding)
    num_classes        : output classes (3: down/stationary/up)
    pe_type            : 'sinusoidal' | 'learnable'
    pooling            : 'mean' | 'last' | 'cls'
                         'cls'  adds a learnable [CLS] token (BERT-style)
    """

    def __init__(
        self,
        n_features: int = 40,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        num_classes: int = 3,
        pe_type: str = "sinusoidal",
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.d_model = d_model

        # Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Optional CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            max_seq_len += 1

        # Positional encoding
        if pe_type == "sinusoidal":
            self.pe: nn.Module = SinusoidalPE(d_model, max_seq_len, dropout)
        elif pe_type == "learnable":
            self.pe = LearnablePE(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Unknown pe_type: {pe_type!r}")

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x                    : (B, T, n_features)
        src_key_padding_mask : (B, T) bool mask — True positions are IGNORED
                               (pass when sequences are padded)

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.size(0)

        # Project features
        x = self.input_proj(x)                              # (B, T, d_model)

        # Prepend CLS token if needed
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)                 # (B, T+1, d_model)
            if src_key_padding_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        # Positional encoding
        x = self.pe(x)                                     # (B, T[+1], d_model)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        if self.pooling == "mean":
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).unsqueeze(-1).float()
                x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                x = x.mean(dim=1)
        elif self.pooling == "last":
            x = x[:, -1, :]
        elif self.pooling == "cls":
            x = x[:, 0, :]                                  # CLS token

        return self.head(x)                                 # (B, num_classes)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_model(cfg: dict, model_type: str = "deeplob") -> nn.Module:
    """
    Build a model from a config dict.

    Parameters
    ----------
    cfg        : dict (typically loaded from configs/model.yaml)
    model_type : 'deeplob' | 'transformer'
    """
    from src.models.deeplob import DeepLOB   # local import to avoid circular

    if model_type == "deeplob":
        mc = cfg.get("deeplob", {})
        return DeepLOB(
            n_features=mc.get("input_levels", 10) * 4,
            num_classes=mc.get("num_classes", 3),
            conv_filters=mc.get("conv_filters", [32, 32, 32]),
            inception_out=mc.get("inception_filters", 64),
            lstm_hidden=mc.get("lstm_hidden", 64),
            lstm_layers=mc.get("lstm_layers", 2),
            dropout=mc.get("dropout", 0.2),
        )

    if model_type == "transformer":
        mc = cfg.get("transformer", {})
        return LOBTransformer(
            n_features=mc.get("input_features", 40),
            d_model=mc.get("d_model", 128),
            nhead=mc.get("nhead", 8),
            num_encoder_layers=mc.get("num_encoder_layers", 4),
            dim_feedforward=mc.get("dim_feedforward", 256),
            dropout=mc.get("dropout", 0.1),
            max_seq_len=mc.get("max_seq_len", 100),
            num_classes=mc.get("num_classes", 3),
        )

    raise ValueError(f"Unknown model type: {model_type!r}")
