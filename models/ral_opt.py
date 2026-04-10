from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_models import NodeFeatureBuilder, LocalGNN, MLP, TransConv


def _infer_task(y_key: str) -> str:
    if y_key in ("slack_eco", "slack"):
        return "reg"
    raise ValueError(f"Unknown y_key='{y_key}'. Supported: 'slack_eco', 'slack'.")


def _mean_pool(g: Any, h: torch.Tensor, *, hetero: bool, ntype: str = "pin") -> torch.Tensor:
    import dgl
    if not hetero:
        g.ndata["_tmp_h"] = h
        out = dgl.mean_nodes(g, "_tmp_h")
        del g.ndata["_tmp_h"]
        return out
    g.nodes[ntype].data["_tmp_h"] = h
    out = dgl.mean_nodes(g, "_tmp_h", ntype=ntype)
    del g.nodes[ntype].data["_tmp_h"]
    return out


class TeacherConeSGFormer(nn.Module):
    """Teacher: SGFormer-style node predictor on a logic cone.

    Returns node predictions, normalized embeddings, outpin embeddings,
    and raw hidden states for downstream distillation.
    """

    def __init__(
        self,
        *,
        hetero: bool,
        x_keys: List[str],
        maxType: int,
        max_size: int,
        hid_dim: int,
        emb_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        target_ntype: str = "pin",
        trans_num_layers: int = 1,
        trans_num_heads: int = 1,
        local_gnn: str = "gcn",
        gnn_num_layers: int = 3,
        gat_heads: int = 4,
        type_emb_dim: int = 32,
        size_emb_dim: int = 16,
        type_key: str = "type_id",
        size_key: str = "size_id",
    ):
        super().__init__()
        self.hetero = bool(hetero)
        self.target_ntype = str(target_ntype)
        self.hid_dim = int(hid_dim)
        self.emb_dim = int(emb_dim)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)

        self.feat = NodeFeatureBuilder(
            hetero=self.hetero, x_keys=x_keys, maxType=maxType, max_size=max_size,
            type_emb_dim=type_emb_dim, size_emb_dim=size_emb_dim,
            type_key=type_key, size_key=size_key, target_ntype=self.target_ntype,
        )
        self._global_cfg = dict(
            hidden_channels=self.hid_dim, num_layers=int(trans_num_layers),
            num_heads=int(trans_num_heads), dropout=self.dropout,
            use_residual=True, use_act=True,
        )
        self._local_cfg = dict(
            hetero=self.hetero, gnn_type=str(local_gnn), hid_dim=self.hid_dim,
            num_layers=int(gnn_num_layers), dropout=self.dropout,
            target_ntype=self.target_ntype, gat_heads=int(gat_heads),
        )
        self.global_enc: Optional[TransConv] = None
        self.local_enc: Optional[LocalGNN] = None

        self.head = MLP(self.hid_dim * 2, self.out_dim, hidden=self.hid_dim, dropout=self.dropout)
        self.proj = nn.Sequential(
            nn.Linear(self.hid_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def _lazy_build(self, in_dim: int):
        dev = self.head.net[0].weight.device
        if self.global_enc is None:
            self.global_enc = TransConv(in_channels=in_dim, **self._global_cfg).to(dev)
        if self.local_enc is None:
            self.local_enc = LocalGNN(in_dim=in_dim, **self._local_cfg).to(dev)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self, g: Any, *, outpin_nids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_raw = self.feat(g)
        self._lazy_build(int(x_raw.shape[-1]))

        h_global = self.global_enc(x_raw)
        h_local = self.local_enc(g, x_raw)
        h = torch.cat([h_global, h_local], dim=-1)
        node_yhat = self.head(h)
        node_z = F.normalize(self.proj(h), dim=-1)

        if outpin_nids is None:
            outpin_z = node_z.new_zeros((0, node_z.shape[-1]))
        else:
            outpin_z = node_z[outpin_nids.to(node_z.device, dtype=torch.int64)]
        return node_yhat, node_z, outpin_z, h


class Student3HopEncoder(nn.Module):
    """Student: 3-hop chunk encoder producing graph embedding + prediction.

    The embedding serves as the retrieval vector and distillation target.
    """

    def __init__(
        self,
        *,
        hetero: bool,
        gnn_type: str,
        x_keys: List[str],
        maxType: int,
        max_size: int,
        hid_dim: int,
        emb_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        target_ntype: str = "pin",
        gnn_num_layers: int = 3,
        gat_heads: int = 4,
        type_emb_dim: int = 32,
        size_emb_dim: int = 16,
        type_key: str = "type_id",
        size_key: str = "size_id",
    ):
        super().__init__()
        self.hetero = bool(hetero)
        self.target_ntype = str(target_ntype)

        self.feat = NodeFeatureBuilder(
            hetero=self.hetero, x_keys=x_keys, maxType=maxType, max_size=max_size,
            type_emb_dim=type_emb_dim, size_emb_dim=size_emb_dim,
            type_key=type_key, size_key=size_key, target_ntype=self.target_ntype,
        )
        self._local_cfg = dict(
            hetero=self.hetero, gnn_type=str(gnn_type), hid_dim=int(hid_dim),
            num_layers=int(gnn_num_layers), dropout=float(dropout),
            target_ntype=self.target_ntype, gat_heads=int(gat_heads),
        )
        self.local_enc: Optional[LocalGNN] = None

        self.pred = MLP(int(hid_dim), int(out_dim), hidden=int(hid_dim), dropout=float(dropout))
        self.proj = nn.Sequential(
            nn.Linear(int(hid_dim), int(emb_dim)),
            nn.ReLU(),
            nn.Linear(int(emb_dim), int(emb_dim)),
        )

    def _lazy_build_local(self, in_dim: int):
        if self.local_enc is not None:
            return
        self.local_enc = LocalGNN(in_dim=in_dim, **self._local_cfg).to(self.pred.net[0].weight.device)

    def forward(self, g: Any, *, center_nids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_raw = self.feat(g)
        self._lazy_build_local(int(x_raw.shape[-1]))
        h = self.local_enc(g, x_raw)
        hg = _mean_pool(g, h, hetero=self.hetero, ntype=self.target_ntype)
        yhat = self.pred(hg)
        z = F.normalize(self.proj(hg), dim=-1)
        return yhat, z
