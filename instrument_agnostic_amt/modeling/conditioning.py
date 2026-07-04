from __future__ import annotations

import torch
import torch.nn as nn


class InstrumentConditioner(nn.Module):
    """Embed the target instrument class used to condition AMT decoding."""

    def __init__(
        self,
        *,
        num_instruments: int,
        dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_instruments <= 0:
            raise ValueError("num_instruments must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.num_instruments = int(num_instruments)
        self.dim = int(dim)
        self.embedding = nn.Embedding(self.num_instruments, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, instrument_ids: torch.Tensor) -> torch.Tensor:
        if instrument_ids.ndim != 1:
            raise ValueError("instrument_ids must have shape [B]")
        if torch.any(instrument_ids < 0) or torch.any(
            instrument_ids >= self.num_instruments
        ):
            raise ValueError("instrument_ids contains an out-of-range class id")
        return self.dropout(self.norm(self.embedding(instrument_ids.long())))
