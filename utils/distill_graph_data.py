from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable

import torch
from torch.utils.data import Dataset

from data.Chunk_Store import ChunkLayout, chunk_to_dgl


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _build_3hop_center_map(chunk_dir: str) -> Dict[Tuple[str, str], str]:
    """
    Build mapping:
      (design_id, center_name(outpin)) -> signature
    from chunk_dir/index.jsonl.
    """
    idx_path = os.path.join(chunk_dir, "index.jsonl")
    rows = _read_jsonl(idx_path)
    m: Dict[Tuple[str, str], str] = {}
    for r in rows:
        sig = r.get("signature")
        meta = r.get("meta", {}) or {}
        did = meta.get("design_id")
        cname = meta.get("center_name")
        if sig and did and cname:
            m[(str(did), str(cname))] = str(sig)
    return m


@dataclass
class ConeDistillSample:
    # teacher input
    g_cone: Any
    y_cone_all: torch.Tensor                # (N, ...) CPU tensor
    cone_meta: Dict[str, Any]
    cone_signature: str

    # outpins inside this cone
    outpin_names: List[str]
    outpin_nids: List[int]          # node ids (within this cone graph, before batching)

    # student inputs (3-hop)
    g_3hop_list: List[Any]                  # aligned with outpin_names
    center_nids_3hop: List[int]             # aligned with outpin_names / g_3hop_list
    y_outpin_list: List[torch.Tensor]       # aligned with outpin_names


class Cone2Outpin3HopDataset(Dataset):
    """
    One item = one logic cone (endpoint fanin cone).
    It returns the cone graph (teacher input) and, for each outpin inside that cone,
    the corresponding 3-hop chunk graph (student input).

    Key: norm stats are NOT computed inside the dataset. They must be provided
         (train-only norm stats) to ensure correct experimental protocol.
    """

    def __init__(
        self,
        *,
        ep_dir: str,
        chunk_dir: str,
        cone_signatures: Optional[Iterable[str]] = None,
        hetero: bool = True,
        build_undirected_hops: bool = True,
        device: Optional[Any] = None,
        normalize: bool = True,
        norm_fields_node: Optional[List[str]] = None,
        norm_fields_edge: Optional[List[str]] = None,
        norm_stats_cone: Optional[Dict[str, Any]] = None,
        norm_stats_3hop: Optional[Dict[str, Any]] = None,
        y_key: str = "slack_eco",
        outpin_key: str = "is_outpin",
    ):
        super().__init__()
        self.ep_dir = str(ep_dir)
        self.chunk_dir = str(chunk_dir)
        self.hetero = bool(hetero)
        self.build_undirected_hops = bool(build_undirected_hops)
        self.device = device
        self.normalize = bool(normalize)
        self.norm_fields_node = norm_fields_node
        self.norm_fields_edge = norm_fields_edge
        self.ns_cone = norm_stats_cone
        self.ns_3hop = norm_stats_3hop
        self.y_key = str(y_key)
        self.outpin_key = str(outpin_key)

        if cone_signatures is None:
            rows = _read_jsonl(os.path.join(self.ep_dir, "index.jsonl"))
            self.cone_sigs = [r["signature"] for r in rows if "signature" in r]
        else:
            self.cone_sigs = [str(s) for s in cone_signatures]

        self.center_map_3hop = _build_3hop_center_map(self.chunk_dir)

        self.layout_ep = ChunkLayout(self.ep_dir)
        self.layout_3h = ChunkLayout(self.chunk_dir)

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

    def _payload_to_graph_3hop(self, payload: Dict[str, Any]) -> Any:
        return chunk_to_dgl(
            payload,
            hetero=self.hetero,
            build_undirected_hops=self.build_undirected_hops,
            device=self.device,
            normalize=self.normalize,
            norm_stats=self.ns_3hop,
            norm_fields_node=self.norm_fields_node,
            norm_fields_edge=self.norm_fields_edge,
        )

    def __getitem__(self, idx: int) -> ConeDistillSample:
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

        y_cone_all = node_feat[self.y_key]  # (N, ...) CPU
        is_outpin = node_feat[self.outpin_key].to(torch.int64).view(-1)

        outpin_names: List[str] = []
        outpin_nids: List[int] = []
        g3_list: List[Any] = []
        center_nids_3hop: List[int] = []
        y_out_list: List[torch.Tensor] = []

        # enumerate outpins in this cone
        for nid in range(len(node_names)):
            if int(is_outpin[nid].item()) != 1:
                continue
            op = str(node_names[nid])

            sig3 = self.center_map_3hop.get((design_id, op), None)
            if sig3 is None:
                # this outpin has no 3-hop chunk stored -> skip
                continue

            payload3 = torch.load(self.layout_3h.chunk_path(sig3), map_location="cpu", weights_only=False)
            g3 = self._payload_to_graph_3hop(payload3)
            center_name_3hop = str(payload3["storage_key"]["center_name"])
            node_names_3hop = list(payload3["node_names"])
            try:
                center_nid_3hop = int(node_names_3hop.index(center_name_3hop))
            except ValueError as e:
                raise RuntimeError(
                    f"3-hop chunk center_name not found in node_names: sig={sig3}, center={center_name_3hop}"
                ) from e

            # outpin label: use cone node label at nid (consistent with your cone node->label storage)
            yop = y_cone_all[nid]
            if not torch.is_tensor(yop):
                yop = torch.tensor(yop)

            outpin_names.append(op)
            outpin_nids.append(nid)
            g3_list.append(g3)
            center_nids_3hop.append(center_nid_3hop)
            y_out_list.append(yop)

        g_cone = self._payload_to_graph_cone(payload_cone)

        return ConeDistillSample(
            g_cone=g_cone,
            y_cone_all=y_cone_all,
            cone_meta=meta,
            cone_signature=sig_cone,
            outpin_names=outpin_names,
            outpin_nids=outpin_nids,
            g_3hop_list=g3_list,
            center_nids_3hop=center_nids_3hop,
            y_outpin_list=y_out_list,
        )