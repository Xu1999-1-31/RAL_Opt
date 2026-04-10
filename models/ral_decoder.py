from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RetrievalHit:
    row: int
    signature: str
    design_id: Optional[str]
    center_name: Optional[str]
    score: float


class TorchRetrievalIndex:
    """Cosine-similarity retrieval index for embeddings saved by build_retrieval_index.

    Files under index_dir:
      - embeddings.pt: {"embeddings": (M, D), ...}
      - values.pt:     {"values": (M, O), ...}
      - mapping.jsonl:  per-row metadata (signature, design_id, ...)
    """

    def __init__(self, *, embeddings: torch.Tensor, values: torch.Tensor, mapping: List[Dict[str, Any]]):
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be 2D (M, D)")
        if values.dim() != 2 or values.shape[0] != embeddings.shape[0]:
            raise ValueError("values must be 2D (M, O) with same M as embeddings")
        self.emb = embeddings.float().contiguous()
        self.val = values.float().contiguous()
        self.mapping = mapping
        self.emb_norm = F.normalize(self.emb, dim=-1)

        M = self.emb.shape[0]
        did_to_rows: Dict[str, List[int]] = {}
        for i, r in enumerate(mapping):
            did = str(r.get("design_id", ""))
            did_to_rows.setdefault(did, []).append(i)
        self._design_keep_mask: Dict[str, torch.Tensor] = {}
        for did, rows in did_to_rows.items():
            mask = torch.ones(M, dtype=torch.bool)
            mask[torch.tensor(rows, dtype=torch.long)] = False
            self._design_keep_mask[did] = mask

    @classmethod
    def load(cls, *, index_dir: str, device: str = "cpu") -> "TorchRetrievalIndex":
        import json, os
        e = torch.load(os.path.join(index_dir, "embeddings.pt"), map_location="cpu", weights_only=False)
        v = torch.load(os.path.join(index_dir, "values.pt"), map_location="cpu", weights_only=False)
        mapping: List[Dict[str, Any]] = []
        with open(os.path.join(index_dir, "mapping.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    mapping.append(json.loads(line))
        obj = cls(embeddings=e["embeddings"], values=v["values"], mapping=mapping)
        obj.to(device)
        return obj

    def to(self, device: str) -> None:
        dev = torch.device(device)
        self.emb = self.emb.to(dev)
        self.val = self.val.to(dev)
        self.emb_norm = self.emb_norm.to(dev)
        self._design_keep_mask = {did: m.to(dev) for did, m in self._design_keep_mask.items()}

    def _apply_exclude_mask(self, scores: torch.Tensor, exclude_design_id: Optional[str]) -> torch.Tensor:
        if exclude_design_id is None:
            return scores
        keep = self._design_keep_mask.get(str(exclude_design_id), None)
        if keep is None:
            return scores
        return scores.masked_fill(~keep, float("-inf"))

    @torch.no_grad()
    def search(
        self, *, query: torch.Tensor, topk: int, exclude_design_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[RetrievalHit]]:
        if query.dim() != 1:
            raise ValueError("query must be 1D (D,)")
        q = F.normalize(query.float().to(self.emb_norm.device), dim=-1)
        scores = torch.mv(self.emb_norm, q)
        scores = self._apply_exclude_mask(scores, exclude_design_id)

        k = min(int(topk), int(scores.numel()))
        if k <= 0:
            return self.emb[:0], self.val[:0], []

        topv, topi = torch.topk(scores, k=k, largest=True, sorted=True)
        hits: List[RetrievalHit] = []
        for ri, sc in zip(topi.detach().cpu().tolist(), topv.detach().cpu().tolist()):
            m = self.mapping[ri] if ri < len(self.mapping) else {}
            hits.append(RetrievalHit(
                row=int(m.get("row", ri)), signature=str(m.get("signature", "")),
                design_id=m.get("design_id"), center_name=m.get("center_name"), score=float(sc),
            ))
        return self.emb[topi], self.val[topi], hits

    @torch.no_grad()
    def search_batch(
        self, *, queries: torch.Tensor, topk: int, exclude_design_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch search: (K, D) queries -> (K, topk, D) embeddings, (K, topk, O) values, (K, topk) scores."""
        if queries.dim() != 2:
            raise ValueError("queries must be (K, D)")
        K = queries.shape[0]
        q = F.normalize(queries.float().to(self.emb_norm.device), dim=-1)
        scores = q @ self.emb_norm.T

        if exclude_design_id is not None:
            keep = self._design_keep_mask.get(str(exclude_design_id), None)
            if keep is not None:
                scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

        k = min(int(topk), scores.shape[1])
        topv, topi = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)

        if k < topk:
            pad = topk - k
            topi = torch.cat([topi, topi.new_zeros((K, pad))], dim=1)
            topv = torch.cat([topv, topv.new_full((K, pad), -1e9)], dim=1)

        return self.emb[topi], self.val[topi], topv


class RALCrossAttnDecoder(nn.Module):
    """Lightweight cross-attention residual decoder for RAL.

    Architecture:
      1. Retrieval aggregation: per-outpin multi-head cross-attention over
         R retrieved neighbors.
      2. Gated outpin fusion: learned gate controls retrieval signal injection.
      3. Cone GNN propagation: GCN/GAT on the original cone graph.
      4. Residual MLP head: out = teacher_yhat + delta.

    The head's last linear is zero-initialised so the model starts at
    teacher_yhat + 0. GNN and cross-attn are lazily built on first forward.
    """

    def __init__(
        self,
        *,
        cone_dim: int,
        emb_dim: int,
        out_dim: int,
        retr_val_dim: Optional[int] = None,
        task: str,
        hid_dim: int = 256,
        dropout: float = 0.1,
        cross_attn_heads: int = 4,
        gnn_type: str = "gcn",
        gnn_num_layers: int = 2,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.cone_dim = int(cone_dim)
        self.emb_dim = int(emb_dim)
        self.out_dim = int(out_dim)
        self.retr_val_dim = int(retr_val_dim) if retr_val_dim is not None else int(out_dim)
        self.task = str(task)
        self.hid_dim = int(hid_dim)
        self.dropout = float(dropout)
        self.cross_attn_heads = int(cross_attn_heads)

        retr_in_dim = self.emb_dim + self.retr_val_dim + 1
        self.retr_proj = nn.Sequential(
            nn.Linear(retr_in_dim, self.hid_dim), nn.ReLU(),
            nn.Dropout(self.dropout), nn.Linear(self.hid_dim, self.hid_dim),
        )
        self.query_proj = nn.Linear(self.cone_dim, self.hid_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hid_dim, num_heads=self.cross_attn_heads,
            dropout=self.dropout, batch_first=True,
        )

        self.gate_net = nn.Sequential(
            nn.Linear(self.cone_dim + self.hid_dim, self.hid_dim), nn.ReLU(),
            nn.Linear(self.hid_dim, self.cone_dim), nn.Sigmoid(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(self.hid_dim, self.cone_dim), nn.ReLU(),
            nn.Linear(self.cone_dim, self.cone_dim),
        )

        self._gnn_cfg: Dict[str, Any] = dict(
            hetero=False, gnn_type=gnn_type, hid_dim=self.hid_dim,
            num_layers=gnn_num_layers, dropout=self.dropout, gat_heads=gat_heads,
        )
        self.cone_gnn: Optional[nn.Module] = None

        self.head = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim), nn.ReLU(),
            nn.Dropout(self.dropout), nn.Linear(self.hid_dim, self.out_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _lazy_build_gnn(self, in_dim: int) -> None:
        if self.cone_gnn is not None:
            return
        from models.base_models import LocalGNN
        dev = self.head[0].weight.device
        self.cone_gnn = LocalGNN(in_dim=in_dim, **self._gnn_cfg).to(dev)

    def forward(
        self,
        *,
        h_cone: torch.Tensor,
        g_cone: Any,
        outpin_nids: torch.Tensor,
        z_retr: torch.Tensor,
        o_retr: torch.Tensor,
        retr_score: Optional[torch.Tensor] = None,
        noise_std: float = 0.0,
        teacher_yhat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N, C = h_cone.shape
        self._lazy_build_gnn(C)
        dev = h_cone.device
        outpin_nids = outpin_nids.to(dev, dtype=torch.long)
        K = int(outpin_nids.shape[0])
        R = int(z_retr.shape[1]) if (z_retr.dim() == 3 and K > 0) else 0

        if noise_std > 0 and K > 0:
            z_retr = z_retr + torch.randn_like(z_retr) * noise_std

        h = h_cone.clone()

        if K > 0 and R > 0:
            if retr_score is None:
                retr_score = z_retr.new_ones((K, R))
            else:
                retr_score = retr_score.to(dev, dtype=z_retr.dtype)

            retr_in = torch.cat([z_retr, o_retr, retr_score.unsqueeze(-1)], dim=-1)
            kv = self.retr_proj(retr_in)
            q = self.query_proj(h_cone[outpin_nids]).unsqueeze(1)
            retr_ctx, _ = self.cross_attn(q, kv, kv)
            retr_ctx = retr_ctx.squeeze(1)

            gate_in = torch.cat([h_cone[outpin_nids], retr_ctx], dim=-1)
            gate = self.gate_net(gate_in)
            update = self.update_net(retr_ctx)
            h[outpin_nids] = h_cone[outpin_nids] + gate * update

        g_cone = g_cone.to(dev)
        h = self.cone_gnn(g_cone, h)

        delta = self.head(h)
        base = teacher_yhat.to(dev, dtype=delta.dtype) if teacher_yhat is not None else torch.zeros_like(delta)
        return base + delta
