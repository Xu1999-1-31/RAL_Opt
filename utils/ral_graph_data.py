from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from data.Chunk_Store import ChunkLayout, chunk_to_dgl


@dataclass
class ConeQuerySample:
    g_cone: Any
    design_id: str
    node_names: List[str]
    outpin_nids: torch.Tensor
    y_cone_all: torch.Tensor


class ConeQueryDataset(Dataset):
    """
    One item = one logic cone (endpoint fanin cone).

    Returns:
      - g_cone: DGL graph
      - design_id: str
      - node_names: list[str] aligned with pin node ids
      - outpin_nids: (K,) int64 tensor of outpin node ids (pin node space)
      - y_cone_all: (N, ...) tensor of labels for ALL pin nodes
    """

    def __init__(
        self,
        *,
        ep_dir: str,
        cone_signatures: Optional[Iterable[str]] = None,
        hetero: bool = True,
        build_undirected_hops: bool = True,
        device: Optional[Any] = None,
        normalize: bool = True,
        norm_stats_cone: Optional[Dict[str, Any]] = None,
        norm_fields_node: Optional[List[str]] = None,
        norm_fields_edge: Optional[List[str]] = None,
        y_key: str = "slack_eco",
        outpin_key: str = "is_outpin",
    ):
        super().__init__()
        self.ep_dir = str(ep_dir)
        self.hetero = bool(hetero)
        self.build_undirected_hops = bool(build_undirected_hops)
        self.device = device
        self.normalize = bool(normalize)
        self.ns_cone = norm_stats_cone or {}
        self.norm_fields_node = norm_fields_node
        self.norm_fields_edge = norm_fields_edge
        self.y_key = str(y_key)
        self.outpin_key = str(outpin_key)

        self.layout_ep = ChunkLayout(self.ep_dir)

        if cone_signatures is None:
            idx_path = os.path.join(self.ep_dir, "index.jsonl")
            sigs: List[str] = []
            if os.path.exists(idx_path):
                import json
                with open(idx_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        s = r.get("signature", None)
                        if s is not None:
                            sigs.append(str(s))
            self.cone_sigs = sigs
        else:
            self.cone_sigs = [str(s) for s in cone_signatures]

    def __len__(self) -> int:
        return len(self.cone_sigs)

    def _payload_to_graph_cone(self, payload: Dict[str, Any]) -> Any:
        return chunk_to_dgl(
            payload,
            hetero=self.hetero,
            build_undirected_hops=self.build_undirected_hops,
            device=self.device,
            normalize=self.normalize,
            norm_stats=self.ns_cone,
            norm_fields_node=self.norm_fields_node,
            norm_fields_edge=self.norm_fields_edge,
        )

    def __getitem__(self, idx: int) -> ConeQuerySample:
        sig_cone = self.cone_sigs[idx]
        payload_cone = torch.load(self.layout_ep.chunk_path(sig_cone), map_location="cpu", weights_only=False)
        meta = payload_cone.get("meta", {}) or {}

        design_id = meta.get("design_id", None)
        if design_id is None:
            sk = payload_cone.get("storage_key", {}) or {}
            design_id = sk.get("design_id", "unknown_design")
        design_id = str(design_id)

        if "node_names" not in payload_cone:
            raise RuntimeError("Logic cone payload must include 'node_names'. Export cones with write_node_names=True.")

        node_names: List[str] = list(payload_cone["node_names"])
        node_feat = payload_cone.get("node_feat", {})
        if self.y_key not in node_feat:
            raise KeyError(f"y_key='{self.y_key}' not found in cone node_feat.")
        if self.outpin_key not in node_feat:
            raise KeyError(f"outpin_key='{self.outpin_key}' not found in cone node_feat.")

        y_cone_all = node_feat[self.y_key]
        is_outpin = node_feat[self.outpin_key].to(torch.int64).view(-1)

        outpin_nids: List[int] = []
        for nid in range(len(node_names)):
            if int(is_outpin[nid].item()) == 1:
                outpin_nids.append(int(nid))

        g_cone = self._payload_to_graph_cone(payload_cone)

        return ConeQuerySample(
            g_cone=g_cone,
            design_id=design_id,
            node_names=node_names,
            outpin_nids=torch.tensor(outpin_nids, dtype=torch.int64),
            y_cone_all=y_cone_all,
        )
