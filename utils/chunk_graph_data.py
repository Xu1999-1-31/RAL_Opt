from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from data.Chunk_Store import (
    chunk_dir,
    ep_dir,
    load_chunk,
    load_all_signatures,
    chunk_to_dgl,
    load_index,
    compute_and_save_norm_stats,
    load_norm_stats,
    compile_norm_stats,
)


# -----------------------------------------------------------------------------
# Unified loading helpers
# -----------------------------------------------------------------------------
def _load_all_signatures_from_index(out_dir: str) -> List[str]:
    """
    Load signatures from out_dir/index.jsonl in file order.
    This is used when out_dir is not the canonical chunk_dir.
    """
    index_path = os.path.join(out_dir, "index.jsonl")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.jsonl not found: {index_path}")

    import json

    sigs: List[str] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sigs.append(row["signature"])
    return sigs


def _load_chunk_from_index(
    out_dir: str,
    signature: str,
    *,
    index_table: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Load one payload using the relpath recorded in index.jsonl.
    If relpath is missing, fall back to chunks/<bucket>/<signature>.pt.
    """
    table = index_table
    if table is None:
        table = load_index(os.path.join(out_dir, "index.jsonl"))

    relpath = None
    row = table.get(signature)
    if row is not None:
        relpath = row.get("relpath")

    if relpath:
        p = os.path.join(out_dir, relpath)
    else:
        bucket = signature[:2]
        p = os.path.join(out_dir, "chunks", bucket, f"{signature}.pt")

    return torch.load(p, map_location="cpu", weights_only=False)


def _load_all_signatures_unified(*, out_dir: str) -> List[str]:
    """
    If out_dir equals chunk_dir, use Chunk_Store APIs.
    Otherwise, read from index.jsonl.
    """
    if out_dir == chunk_dir:
        return list(load_all_signatures())
    return _load_all_signatures_from_index(out_dir)


def _load_chunk_unified(
    signature: str,
    *,
    out_dir: str,
    index_table: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    If out_dir equals chunk_dir, use Chunk_Store APIs.
    Otherwise, load from index.jsonl + relpath.
    """
    if out_dir == chunk_dir:
        return load_chunk(signature)
    return _load_chunk_from_index(out_dir, signature, index_table=index_table)


def _get_node_data(g: Any, *, hetero: bool, ntype: str) -> Dict[str, torch.Tensor]:
    """
    Return node data dict for homo or hetero graphs.
    """
    if hetero:
        return g.nodes[ntype].data
    return g.ndata


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def signatures_by_designs(
    design_ids: Sequence[str],
    *,
    out_dir: str = chunk_dir,
) -> List[str]:
    """
    Filter signatures by meta.design_id recorded in index.jsonl.
    """
    index_path = os.path.join(out_dir, "index.jsonl")
    table = load_index(index_path)

    want = set(design_ids)
    out: List[str] = []
    for sig, row in table.items():
        meta = row.get("meta", {})
        did = meta.get("design_id", None)
        if did in want:
            out.append(sig)
    return out


def compute_per_design_label_stats(
    design_ids: Sequence[str],
    *,
    out_dir: str = chunk_dir,
    y_feat_key: str = "slack",
    y_ntype: str = "pin",
    y_select: str = "all",
    label_norm_ref_key: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-design label mean/std for regression normalization.

    Returns ``{design_id: {"mean": float, "std": float, "n": int}}``.
    Only designs in *design_ids* are scanned.

    When *label_norm_ref_key* is given, the statistics are computed from that
    feature instead of *y_feat_key*.  For example, passing
    ``label_norm_ref_key="slack"`` while ``y_feat_key="slack_eco"`` uses the
    pre-ECO slack distribution (which is already known) to normalize the
    post-ECO slack labels.

    Results are cached to ``<out_dir>/label_norm_stats_<ref_key>.pt``.  On
    subsequent calls the cache is returned directly when the sorted design
    list matches.
    """
    ref_key = label_norm_ref_key if label_norm_ref_key is not None else y_feat_key
    sorted_designs = sorted(design_ids)

    cache_path = os.path.join(out_dir, f"label_norm_stats_{ref_key}.pt")
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("_designs") == sorted_designs:
            print(f"[label_norm] Cache hit for {cache_path} — designs match, skipping recomputation.")
            return cached["stats"]

    index_path = os.path.join(out_dir, "index.jsonl")
    table = load_index(index_path)

    design_sums: Dict[str, float] = {}
    design_sq_sums: Dict[str, float] = {}
    design_counts: Dict[str, int] = {}

    want = set(design_ids)
    n_scanned = 0
    for sig, row in table.items():
        meta = row.get("meta", {})
        did = meta.get("design_id", None)
        if did not in want:
            continue

        payload = _load_chunk_unified(sig, out_dir=out_dir, index_table=table)
        nf = payload.get("node_feat", {})
        if ref_key not in nf:
            continue
        feat = nf[ref_key].float().view(-1)
        valid = torch.isfinite(feat) & (feat != -1.0)
        if not valid.any():
            continue
        vals = feat[valid]
        design_sums[did] = design_sums.get(did, 0.0) + float(vals.sum().item())
        design_sq_sums[did] = design_sq_sums.get(did, 0.0) + float((vals * vals).sum().item())
        design_counts[did] = design_counts.get(did, 0) + int(vals.numel())
        n_scanned += 1
    print(f"[label_norm] scanned {n_scanned} chunks for {len(want)} designs (ref_key={ref_key!r})")

    stats: Dict[str, Dict[str, float]] = {}
    for did in design_ids:
        n = design_counts.get(did, 0)
        if n < 2:
            stats[did] = {"mean": 0.0, "std": 1.0, "n": n}
            continue
        mean = design_sums[did] / n
        var = max(design_sq_sums[did] / n - mean * mean, 0.0)
        std = max(var ** 0.5, 1e-8)
        stats[did] = {"mean": mean, "std": std, "n": n}

    torch.save({"_designs": sorted_designs, "stats": stats}, cache_path)
    print(f"[label_norm] Saved label_norm_stats to {cache_path}")
    return stats


def build_train_norm_stats(
    train_signatures: Sequence[str],
    *,
    out_dir: str = chunk_dir,
    fields_node: Optional[List[str]] = None,
    fields_edge: Optional[List[str]] = None,
    designs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute normalization stats on train signatures and return compiled stats.
    It also saves norm_stats.pt under out_dir.

    When *designs* is provided the function first checks whether an existing
    ``norm_stats.pt`` was already computed for the same sorted design list.
    If so, the cached file is reused and recomputation is skipped.
    """
    designs_list: Optional[List[str]] = sorted(designs) if designs is not None else None

    if designs_list is not None:
        try:
            existing = load_norm_stats(out_dir=out_dir)
            cached_designs = existing.get("meta", {}).get("designs")
            if cached_designs is not None and cached_designs == designs_list:
                print(f"[norm_stats] Cache hit for {out_dir} — designs match, skipping recomputation.")
                return compile_norm_stats(existing)
        except FileNotFoundError:
            pass

    compute_and_save_norm_stats(
        out_dir=out_dir,
        signatures=list(train_signatures),
        fields_node=fields_node,
        fields_edge=fields_edge,
        designs=designs_list,
    )
    ns = load_norm_stats(out_dir=out_dir)
    return compile_norm_stats(ns)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ChunkGraphDataset(Dataset):
    """
    A dataset that loads chunk payloads, converts them to DGL graphs, and extracts y.

    Supported y_select modes:
      center: select center node y as a (1, ...) tensor
      names: select a list of nodes by names and return (K, ...) tensor
      ids: select a list of local node ids and return (K, ...) tensor
      mask: select nodes whose mask key equals y_mask_value and return (K, ...) tensor
      all: select all nodes of y_ntype and return (N, ...) tensor
      none: do not extract y and return None

    Output item dict fields:
      g: DGLGraph or DGLHeteroGraph
      y: Tensor or None
      y_info: dict describing selection
      signature: str
      center_name: str
      meta: dict
    """

    def __init__(
        self,
        *,
        out_dir: str = chunk_dir,
        signatures: Optional[Sequence[str]] = None,
        hetero: bool = False,
        build_undirected_hops: bool = True,
        device: Optional[Any] = None,
        cache_graph: bool = False,
        return_payload: bool = False,
        normalize: bool = False,
        norm_stats: Optional[Dict[str, Any]] = None,
        norm_fields_node: Optional[List[str]] = None,
        norm_fields_edge: Optional[List[str]] = None,
        y_feat_key: str = "slack",
        y_ntype: str = "pin",
        y_select: str = "center",
        y_node_names: Optional[Sequence[str]] = None,
        y_node_ids: Optional[Sequence[int]] = None,
        y_node_mask_key: Optional[str] = None,
        y_mask_value: int = 1,
    ):
        super().__init__()
        self.out_dir = out_dir
        self._index_table = load_index(os.path.join(out_dir, "index.jsonl"))

        self.hetero = hetero
        self.build_undirected_hops = build_undirected_hops
        self.device = device
        self.cache_graph = cache_graph
        self.return_payload = return_payload

        self.normalize = normalize
        self.norm_stats = norm_stats
        self.norm_fields_node = norm_fields_node
        self.norm_fields_edge = norm_fields_edge

        self.y_feat_key = y_feat_key
        self.y_ntype = y_ntype
        self.y_select = y_select
        self.y_node_names = list(y_node_names) if y_node_names is not None else None
        self.y_node_ids = list(y_node_ids) if y_node_ids is not None else None
        self.y_node_mask_key = y_node_mask_key
        self.y_mask_value = y_mask_value

        self.label_norm_stats: Optional[Dict[str, Dict[str, float]]] = None

        if signatures is None:
            self.signatures = _load_all_signatures_unified(out_dir=out_dir)
        else:
            self.signatures = list(signatures)

        self._graph_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.signatures)

    def _select_local_ids(self, payload: Dict[str, Any], g: Any) -> List[int]:
        """
        Compute local node ids to select for y based on y_select mode,
        except for y_select == all and y_select == none.
        """
        if "node_ids" in payload and hasattr(payload["node_ids"], "numel"):
            num_nodes = int(payload["node_ids"].numel())
        else:
            num_nodes = None

        if self.y_select == "center":
            if "node_names" not in payload:
                raise RuntimeError("y_select center requires payload node_names")
            center_name = payload["storage_key"]["center_name"]
            names = payload["node_names"]
            try:
                return [int(names.index(center_name))]
            except ValueError:
                raise RuntimeError(f"center_name not found in payload node_names: {center_name}")

        if self.y_select == "names":
            if self.y_node_names is None:
                raise ValueError("y_select names requires y_node_names")
            if "node_names" not in payload:
                raise RuntimeError("y_select names requires payload node_names")
            name2lid = {n: i for i, n in enumerate(payload["node_names"])}
            return [int(name2lid[n]) for n in self.y_node_names if n in name2lid]

        if self.y_select == "ids":
            if self.y_node_ids is None:
                raise ValueError("y_select ids requires y_node_ids")
            lids: List[int] = []
            for i in self.y_node_ids:
                ii = int(i)
                if num_nodes is not None and (ii < 0 or ii >= num_nodes):
                    continue
                lids.append(ii)
            return lids

        if self.y_select == "mask":
            if self.y_node_mask_key is None:
                raise ValueError("y_select mask requires y_node_mask_key")
            nd = _get_node_data(g, hetero=self.hetero, ntype=self.y_ntype)
            if self.y_node_mask_key not in nd:
                raise KeyError(
                    f"mask key not found: {self.y_node_mask_key}, available: {list(nd.keys())}"
                )
            mask = nd[self.y_node_mask_key].view(-1)
            lids = torch.nonzero(mask == int(self.y_mask_value), as_tuple=False).view(-1).tolist()
            return [int(x) for x in lids]

        raise ValueError(f"Unknown y_select: {self.y_select}")

    def _extract_y(self, payload: Dict[str, Any], g: Any) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Extract y from graph node features using y_feat_key.
        """
        if self.y_select == "none":
            return None, {"mode": "none"}

        nd = _get_node_data(g, hetero=self.hetero, ntype=self.y_ntype)
        if self.y_feat_key not in nd:
            raise KeyError(f"y_feat_key not found: {self.y_feat_key}, available: {list(nd.keys())}")

        feat = nd[self.y_feat_key]

        if self.y_select == "all":
            y = feat
            return y, {"mode": "all", "num": int(y.shape[0]), "feat_key": self.y_feat_key, "ntype": self.y_ntype}

        lids = self._select_local_ids(payload, g)
        if len(lids) == 0:
            return None, {"mode": self.y_select, "num": 0, "feat_key": self.y_feat_key, "ntype": self.y_ntype}

        idx = torch.tensor(lids, dtype=torch.long, device=feat.device)
        y = feat.index_select(0, idx)
        return y, {"mode": self.y_select, "num": len(lids), "feat_key": self.y_feat_key, "ntype": self.y_ntype}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sig = self.signatures[idx]
        payload = _load_chunk_unified(sig, out_dir=self.out_dir, index_table=self._index_table)

        center_name = payload["storage_key"]["center_name"]
        meta = payload.get("meta", {})

        if self.cache_graph and sig in self._graph_cache:
            g = self._graph_cache[sig]
        else:
            g = chunk_to_dgl(
                payload,
                hetero=self.hetero,
                build_undirected_hops=self.build_undirected_hops,
                device=None,
                normalize=self.normalize,
                norm_stats=self.norm_stats,
                norm_fields_node=self.norm_fields_node,
                norm_fields_edge=self.norm_fields_edge,
            )
            if self.cache_graph:
                self._graph_cache[sig] = g

        y, y_info = self._extract_y(payload, g)

        design_id = meta.get("design_id", "")

        if y is not None and self.label_norm_stats is not None and design_id in self.label_norm_stats:
            ls = self.label_norm_stats[design_id]
            y = (y.float() - ls["mean"]) / ls["std"]

        out: Dict[str, Any] = {
            "g": g,
            "y": y,
            "y_info": y_info,
            "signature": sig,
            "center_name": center_name,
            "meta": meta,
            "design_id": design_id,
        }
        if self.return_payload:
            out["payload"] = payload
        return out


# -----------------------------------------------------------------------------
# Collate function and dataloader builders
# -----------------------------------------------------------------------------
def make_chunk_collate_fn():
    """
    Collate a list of dataset items into a batch.

    For y_select modes other than all:
      batch y is a list of tensors, one per graph, possibly with different K.

    For y_select mode all:
      batch y is a single tensor created by concatenating each graph y along dim 0.
      This aligns with DGL batching order for the same node type.

    For y_select mode none:
      batch y is None.
    """

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import dgl

        graphs = [b["g"] for b in batch]
        bg = dgl.batch(graphs)

        ys = [b["y"] for b in batch]
        y_infos = [b["y_info"] for b in batch]

        if all(y is None for y in ys):
            yb = None
        else:
            mode0 = y_infos[0].get("mode", "")
            if mode0 == "all":
                yb = torch.cat(ys, dim=0)
            else:
                yb = ys

        return {
            "g": bg,
            "y": yb,
            "y_info": y_infos,
            "signature": [b["signature"] for b in batch],
            "center_name": [b["center_name"] for b in batch],
            "meta": [b["meta"] for b in batch],
            "design_id": [b.get("design_id", "") for b in batch],
        }

    return _collate


def build_chunk_dataloader(
    *,
    out_dir: str = chunk_dir,
    signatures: Optional[Sequence[str]] = None,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    hetero: bool = False,
    build_undirected_hops: bool = True,
    device: Optional[Any] = None,
    cache_graph: bool = False,
    normalize: bool = False,
    norm_stats: Optional[Dict[str, Any]] = None,
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
    y_feat_key: str = "slack",
    y_ntype: str = "pin",
    y_select: str = "center",
    y_node_names: Optional[Sequence[str]] = None,
    y_node_ids: Optional[Sequence[int]] = None,
    y_node_mask_key: Optional[str] = None,
    y_mask_value: int = 1,
    label_norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[ChunkGraphDataset, DataLoader]:
    """
    Build dataset and dataloader for a signature list.

    label_norm_stats: per-design label z-score stats from
        ``compute_per_design_label_stats``.  When provided the dataset
        normalizes y values on-the-fly: ``y_norm = (y - mean) / std``.
    """
    ds = ChunkGraphDataset(
        out_dir=out_dir,
        signatures=signatures,
        hetero=hetero,
        build_undirected_hops=build_undirected_hops,
        device=device,
        cache_graph=cache_graph,
        normalize=normalize,
        norm_stats=norm_stats,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        y_feat_key=y_feat_key,
        y_ntype=y_ntype,
        y_select=y_select,
        y_node_names=y_node_names,
        y_node_ids=y_node_ids,
        y_node_mask_key=y_node_mask_key,
        y_mask_value=y_mask_value,
    )
    if label_norm_stats is not None:
        ds.label_norm_stats = label_norm_stats

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory and device is not None and str(device).startswith("cuda")),
        collate_fn=make_chunk_collate_fn(),
    )
    return ds, dl


def build_train_test_loaders_by_design(
    *,
    out_dir: str = chunk_dir,
    train_designs: Sequence[str],
    test_designs: Sequence[str],
    batch_size: int = 64,
    hetero: bool = False,
    device: Optional[Any] = None,
    cache_graph: bool = False,
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
    y_feat_key: str = "slack_eco",
    y_ntype: str = "pin",
    y_select: str = "center",
    label_norm: bool = False,
    label_norm_ref_key: Optional[str] = "slack",
) -> Tuple[Dict[str, Any], Tuple[ChunkGraphDataset, DataLoader], Tuple[ChunkGraphDataset, DataLoader], Optional[Dict[str, Dict[str, float]]]]:
    """
    Build train and test loaders, and compute normalization from train set only.

    When *label_norm* is True, per-design label z-score statistics are computed
    from the training designs and applied to both train and test loaders.
    *label_norm_ref_key* controls which feature's distribution is used to
    compute the stats (default ``"slack"``).
    The fourth return value is the label_norm_stats dict (None when disabled).
    """
    train_sigs = signatures_by_designs(train_designs, out_dir=out_dir)
    test_sigs = signatures_by_designs(test_designs, out_dir=out_dir)

    if len(train_sigs) == 0:
        raise RuntimeError("No train signatures found. Check design_id in index.jsonl meta.")
    if len(test_sigs) == 0:
        raise RuntimeError("No test signatures found. Check design_id in index.jsonl meta.")

    ns = build_train_norm_stats(
        train_sigs,
        out_dir=out_dir,
        fields_node=norm_fields_node,
        fields_edge=norm_fields_edge,
        designs=train_designs,
    )

    lns: Optional[Dict[str, Dict[str, float]]] = None
    if label_norm:
        all_designs = list(set(list(train_designs) + list(test_designs)))
        lns = compute_per_design_label_stats(
            all_designs, out_dir=out_dir, y_feat_key=y_feat_key, y_ntype=y_ntype,
            y_select=y_select, label_norm_ref_key=label_norm_ref_key,
        )
        print(f"[label_norm] Per-design label stats:")
        for did, s in lns.items():
            print(f"  {did}: mean={s['mean']:.6f} std={s['std']:.6f} n={s['n']}")

    train_ds, train_dl = build_chunk_dataloader(
        out_dir=out_dir,
        signatures=train_sigs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        hetero=hetero,
        build_undirected_hops=True,
        device=device,
        cache_graph=cache_graph,
        normalize=True,
        norm_stats=ns,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        y_feat_key=y_feat_key,
        y_ntype=y_ntype,
        y_select=y_select,
        label_norm_stats=lns,
    )

    test_ds, test_dl = build_chunk_dataloader(
        out_dir=out_dir,
        signatures=test_sigs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        hetero=hetero,
        build_undirected_hops=True,
        device=device,
        cache_graph=cache_graph,
        normalize=True,
        norm_stats=ns,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        y_feat_key=y_feat_key,
        y_ntype=y_ntype,
        y_select=y_select,
        label_norm_stats=lns,
    )

    return ns, (train_ds, train_dl), (test_ds, test_dl), lns


# -----------------------------------------------------------------------------
# Debug main
# -----------------------------------------------------------------------------
def _print_graph_info(g: Any) -> None:
    import dgl

    if isinstance(g, dgl.DGLHeteroGraph):
        print("Graph type: hetero")
        print("Node types:", g.ntypes)
        print("Edge types:", g.etypes)
        for nt in g.ntypes:
            print("Num nodes", nt, ":", g.num_nodes(nt))
        for et in g.etypes:
            print("Num edges", et, ":", g.num_edges(et))
    else:
        print("Graph type: homo")
        print("Num nodes:", g.num_nodes())
        print("Num edges:", g.num_edges())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=ep_dir, help="chunk store root directory containing index.jsonl")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--hetero", action="store_true")
    ap.add_argument("--homo", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_batches", type=int, default=1)

    ap.add_argument("--y_feat_key", type=str, default="slack_eco")
    ap.add_argument("--y_ntype", type=str, default="pin")
    ap.add_argument(
        "--y_select",
        type=str,
        default="center",
        choices=["center", "names", "ids", "mask", "all", "none"],
    )
    ap.add_argument("--y_mask_key", type=str, default=None)
    ap.add_argument("--y_mask_value", type=int, default=1)
    ap.add_argument("--y_names", type=str, default=None)
    ap.add_argument("--y_ids", type=str, default=None)

    args = ap.parse_args()

    hetero = True
    if args.homo:
        hetero = False
    elif args.hetero:
        hetero = True

    dev = torch.device(args.device)

    y_node_names = None
    if args.y_names:
        y_node_names = [x.strip() for x in args.y_names.split(",") if x.strip()]

    y_node_ids = None
    if args.y_ids:
        y_node_ids = [int(x.strip()) for x in args.y_ids.split(",") if x.strip()]

    ds, dl = build_chunk_dataloader(
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        hetero=hetero,
        device=dev,
        cache_graph=False,
        normalize=False,
        y_feat_key=args.y_feat_key,
        y_ntype=args.y_ntype,
        y_select=args.y_select,
        y_node_names=y_node_names,
        y_node_ids=y_node_ids,
        y_node_mask_key=args.y_mask_key,
        y_mask_value=args.y_mask_value,
    )

    print("Dataset size:", len(ds))
    for bi, batch in enumerate(dl):
        print("Batch index:", bi)
        print("Batch signatures count:", len(batch["signature"]))
        _print_graph_info(batch["g"])
        if batch["y"] is None:
            print("Batch y: None")
        else:
            if isinstance(batch["y"], list):
                print("Batch y is list, length:", len(batch["y"]))
                print("First y shape:", tuple(batch["y"][0].shape) if batch["y"][0] is not None else None)
            else:
                print("Batch y is tensor, shape:", tuple(batch["y"].shape))
        print("First y_info:", batch["y_info"][0])
        if bi + 1 >= args.max_batches:
            break


if __name__ == "__main__":
    main()