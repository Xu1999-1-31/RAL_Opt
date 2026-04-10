from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import GATConv, HeteroGraphConv, GraphConv, GINConv


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP head."""
    def __init__(self, in_dim: int, out_dim: int, *, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodeFeatureBuilder(nn.Module):
    """Build per-node input features.

    Output feature = concat(
        [float features from x_keys],
        [type embedding],
        [size embedding],
    )

    Notes:
      - We map -1 -> 0 for embeddings. Therefore embedding tables use +2 capacity.
      - Works for both homographs and heterographs by reading node data from target_ntype.
    """

    def __init__(
        self,
        *,
        hetero: bool,
        x_keys: List[str],
        maxType: int,
        max_size: int,
        type_emb_dim: int = 32,
        size_emb_dim: int = 16,
        type_key: str = "type_id",
        size_key: str = "size_id",
        target_ntype: str = "pin",
    ):
        super().__init__()
        self.hetero = bool(hetero)
        self.x_keys = list(x_keys)
        self.type_key = str(type_key)
        self.size_key = str(size_key)
        self.target_ntype = str(target_ntype)

        # reserve 0 for unknown (-1 mapped to 0), so +2
        self.type_emb = nn.Embedding(num_embeddings=maxType + 2, embedding_dim=type_emb_dim)
        self.size_emb = nn.Embedding(num_embeddings=max_size + 2, embedding_dim=size_emb_dim)

        self.in_dim: int = -1
        self._cont_dim_cached: Optional[int] = None

    def _get_node_data(self, g: Any) -> Dict[str, torch.Tensor]:
        if self.hetero:
            return g.nodes[self.target_ntype].data
        return g.ndata

    def forward(self, g: Any) -> torch.Tensor:
        nd = self._get_node_data(g)

        # 1) continuous features
        cont_list: List[torch.Tensor] = []
        for k in self.x_keys:
            if k not in nd:
                raise KeyError(f"x_key '{k}' not found in node data. Available keys: {list(nd.keys())}")
            xk = nd[k]
            if xk.dtype != torch.float32:
                xk = xk.float()
            if xk.dim() == 1:
                xk = xk.unsqueeze(-1)
            cont_list.append(xk)
        cont = torch.cat(cont_list, dim=-1) if len(cont_list) > 0 else None

        # 2) embeddings
        if self.type_key not in nd or self.size_key not in nd:
            raise KeyError(f"Need '{self.type_key}' and '{self.size_key}' in node data for embeddings.")
        tid = nd[self.type_key].to(torch.int64).view(-1)
        sid = nd[self.size_key].to(torch.int64).view(-1)

        tid = torch.clamp(tid, min=-1) + 1  # -1->0, 0->1, ...
        sid = torch.clamp(sid, min=-1) + 1

        t_emb = self.type_emb(tid)
        s_emb = self.size_emb(sid)

        emb_parts = [t_emb, s_emb]

        # 3) concat
        if cont is None:
            x = torch.cat(emb_parts, dim=-1)
            cont_dim = 0
        else:
            x = torch.cat([cont] + emb_parts, dim=-1)
            cont_dim = int(cont.shape[-1])

        # cache/validate in_dim
        emb_total = self.type_emb.embedding_dim + self.size_emb.embedding_dim * len(emb_parts[1:])
        if self._cont_dim_cached is None:
            self._cont_dim_cached = cont_dim
            self.in_dim = cont_dim + emb_total
        else:
            expect = self._cont_dim_cached + emb_total
            if x.shape[-1] != expect:
                raise RuntimeError(f"Input dim changed across batches: got {x.shape[-1]}, expected {expect}")

        return x


# --------------------------------------------------------------------------------------
# Local GNN blocks (node-level)
# --------------------------------------------------------------------------------------

def _infer_local_out_dim(hid_dim: int, *, gnn_type: str, gat_heads: int) -> Tuple[int, int]:
    """Return (per_head_dim, num_heads) for GAT, else (hid_dim, 1)."""
    gnn_type = gnn_type.lower()
    if gnn_type == "gat":
        if hid_dim % gat_heads != 0:
            raise ValueError("hid_dim must be divisible by gat_heads")
        return hid_dim // gat_heads, gat_heads
    return hid_dim, 1


class LocalGNN(nn.Module):
    """Selectable local message passing on the graph.

    Supported:
      - gnn_type='gcn' : DGL GraphConv
      - gnn_type='gat' : DGL GATConv
      - gnn_type='gin' : DGL GINConv (homogeneous only)

    Heterograph support:
      - gcn/gat are supported via HeteroGraphConv on the canonical edge types:
            ('pin','cell_arc','pin') and ('pin','net_arc','pin')  (your dataset uses rel names 'cell_arc'/'net_arc')
      - gin is NOT supported for heterographs.
    """

    def __init__(
        self,
        *,
        hetero: bool,
        gnn_type: str,
        in_dim: int,
        hid_dim: int,
        num_layers: int,
        dropout: float,
        target_ntype: str = "pin",
        gat_heads: int = 4,
    ):
        super().__init__()
        self.hetero = bool(hetero)
        self.gnn_type = str(gnn_type).strip().lower()
        self.hid_dim = int(hid_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.target_ntype = str(target_ntype)
        self.gat_heads = int(gat_heads)

        if self.gnn_type not in ("gcn", "gat", "gin"):
            raise ValueError(f"Unknown gnn_type='{gnn_type}'. Use 'gcn', 'gat', or 'gin'.")

        if self.hetero and self.gnn_type == "gin":
            raise ValueError("LocalGNN gnn_type='gin' does not support heterographs.")

        # Project raw x to hidden dim before message passing.
        self.proj = nn.Linear(in_dim, self.hid_dim)

        self.layers = nn.ModuleList()
        if not self.hetero:
            self._build_homo()
        else:
            self._build_hetero()

    def _build_homo(self):
        if self.gnn_type == "gcn":
            for _ in range(self.num_layers):
                self.layers.append(
                    GraphConv(self.hid_dim, self.hid_dim, norm="both", weight=True, bias=True, allow_zero_in_degree=True)
                )
        elif self.gnn_type == "gat":
            per_head, heads = _infer_local_out_dim(self.hid_dim, gnn_type="gat", gat_heads=self.gat_heads)
            for _ in range(self.num_layers):
                self.layers.append(
                    GATConv(
                        self.hid_dim,
                        per_head,
                        heads,
                        feat_drop=self.dropout,
                        attn_drop=self.dropout,
                        residual=True,
                        activation=F.elu,
                        allow_zero_in_degree=True,
                    )
                )
        else:  # gin (homo only)
            def mlp():
                return nn.Sequential(
                    nn.Linear(self.hid_dim, self.hid_dim),
                    nn.ReLU(),
                    nn.Linear(self.hid_dim, self.hid_dim),
                )
            for _ in range(self.num_layers):
                self.layers.append(GINConv(mlp(), "sum"))

    def _build_hetero(self):
        if self.gnn_type == "gcn":
            def hetero_layer():
                mods = {
                    "cell_arc": GraphConv(self.hid_dim, self.hid_dim, norm="both", weight=True, bias=True, allow_zero_in_degree=True),
                    "net_arc": GraphConv(self.hid_dim, self.hid_dim, norm="both", weight=True, bias=True, allow_zero_in_degree=True),
                }
                return HeteroGraphConv(mods, aggregate="mean")
            for _ in range(self.num_layers):
                self.layers.append(hetero_layer())

        elif self.gnn_type == "gat":
            per_head, heads = _infer_local_out_dim(self.hid_dim, gnn_type="gat", gat_heads=self.gat_heads)

            def hetero_layer():
                mods = {
                    "cell_arc": GATConv(self.hid_dim, per_head, heads, feat_drop=self.dropout, attn_drop=self.dropout,
                                       residual=True, activation=F.elu, allow_zero_in_degree=True),
                    "net_arc": GATConv(self.hid_dim, per_head, heads, feat_drop=self.dropout, attn_drop=self.dropout,
                                      residual=True, activation=F.elu, allow_zero_in_degree=True),
                }
                return HeteroGraphConv(mods, aggregate="mean")
            for _ in range(self.num_layers):
                self.layers.append(hetero_layer())

        else:
            raise RuntimeError("Unreachable: hetero+gin is blocked in __init__.")

    def forward(self, g: Any, x_raw: torch.Tensor) -> torch.Tensor:
        x = self.proj(x_raw)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if not self.hetero:
            h = x
            for layer in self.layers:
                h = layer(g, h)
                # GATConv returns (N, heads, per_head) -> flatten to (N, hid_dim)
                if self.gnn_type == "gat":
                    h = h.flatten(1)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            return h

        # hetero: only keep target node type outputs
        h_dict: Dict[str, torch.Tensor] = {self.target_ntype: x}
        for layer in self.layers:
            h_dict = layer(g, h_dict)
            h_dict = {k: v.flatten(1) if (self.gnn_type == "gat") else v for k, v in h_dict.items()}
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()}
        return h_dict[self.target_ntype]


# --------------------------------------------------------------------------------------
# SGFormer-style global module (fast attention, edge-independent)
# --------------------------------------------------------------------------------------

class TransConvLayer(nn.Module):
    """Fast attention (SGFormer-style) over a set of nodes.

    This is edge-independent: it aggregates global information using all-pairs attention
    but with linear complexity.

    Compatibility with heterographs:
      - Works naturally: we apply it to the node feature matrix of the target node type.
      - No edge_index / adjacency is required.
    """

    def __init__(self, in_channels: int, out_channels: int, num_heads: int, use_weight: bool = True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads) if use_weight else None

        self.out_channels = int(out_channels)
        self.num_heads = int(num_heads)
        self.use_weight = bool(use_weight)

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.Wv is not None:
            self.Wv.reset_parameters()

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, query_input: torch.Tensor, source_input: torch.Tensor) -> torch.Tensor:
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        qs = qs / (torch.norm(qs, p=2, keepdim=True) + 1e-6)
        ks = ks / (torch.norm(ks, p=2, keepdim=True) + 1e-6)
        N = int(qs.shape[0])

        # numerator
        kvs = torch.einsum("nhm,nhd->hmd", ks, vs)               # [H, M, D]
        attention_num = torch.einsum("lhm,hmd->lhd", qs, kvs)    # [N, H, D]
        attention_num = attention_num + N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]], device=ks.device, dtype=ks.dtype)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)         # [H, M]
        attention_normalizer = torch.einsum("lhm,hm->lh", qs, ks_sum)  # [N, H]
        attention_normalizer = attention_normalizer.unsqueeze(-1) + N

        attn_output = attention_num / (attention_normalizer + 1e-6)  # [N, H, D]
        return attn_output.mean(dim=1)  # [N, D]


class TransConv(nn.Module):
    """Stacked TransConvLayer with input projection and residual."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)
        self.use_act = bool(use_act)

        self.in_fc = nn.Linear(in_channels, hidden_channels)
        self.in_ln = nn.LayerNorm(hidden_channels)

        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        per_head = hidden_channels // num_heads

        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(TransConvLayer(hidden_channels, per_head, num_heads=num_heads, use_weight=True))
            self.lns.append(nn.LayerNorm(hidden_channels))

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_fc(x)
        x = self.in_ln(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        prev = x
        for layer, ln in zip(self.layers, self.lns):
            y = layer(x, x)
            if self.use_residual:
                y = (y + prev) / 2.0
            y = ln(y)
            if self.use_act:
                y = self.act(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            prev = y
            x = y
        return x