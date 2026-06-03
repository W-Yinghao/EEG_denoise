"""Learnable subject embeddings (handoff §1.2, §4).

Each of the ``N`` training subjects has a ``d``-dim embedding (N=9, d=128). An extra slot at index
``N`` holds a null/averaged embedding used for the ``--no_subject`` ablation and as the default
when no subject id is supplied ([DD-4]).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SubjectEmbedding(nn.Module):
    """Embedding table over ``num_subjects`` subjects plus one null slot."""

    def __init__(self, num_subjects: int, emb_dim: int) -> None:
        super().__init__()
        self.num_subjects = num_subjects
        self.null_index = num_subjects
        self.embed = nn.Embedding(num_subjects + 1, emb_dim)

    def forward(self, subject_ids: Tensor) -> Tensor:
        """Look up embeddings for ``(B,)`` 0-based subject ids (or ``null_index`` for null)."""
        return self.embed(subject_ids)

    def null_ids(self, batch_size: int, device: torch.device) -> Tensor:
        """Return a ``(B,)`` tensor of the null-embedding index (no-subject conditioning)."""
        return torch.full((batch_size,), self.null_index, dtype=torch.long, device=device)
