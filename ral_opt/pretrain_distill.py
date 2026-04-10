from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.env_setup import setup_env
from utils.distill_graph_data import Cone2Outpin3HopDataset
from models.ral_opt import TeacherConeSGFormer, Student3HopEncoder, _infer_task

from data.Chunk_Store import ChunkLayout, chunk_to_dgl
from utils.chunk_graph_data import signatures_by_designs, build_train_norm_stats
from data.Data_var import ep_dir, chunk_dir
from ral_opt.ral_var import out_dir, student_ckpt_path, teacher_ckpt_path
import tqdm

setup_env()

logger = logging.getLogger("RAL_Opt.pretrain_distill")


def _num_target_nodes(g: Any, *, hetero: bool, target_ntype: str = "pin") -> int:
    if not hetero:
        return int(g.num_nodes())
    return int(g.num_nodes(target_ntype))


def _batch_outpin_nids(batch: List[Any], *, hetero: bool, target_ntype: str = "pin") -> torch.Tensor:
    """Build a 1D tensor of outpin node ids on the *batched* cone graph."""
    nids_all: List[torch.Tensor] = []
    offset = 0
    for it in batch:
        if not hasattr(it, "outpin_nids"):
            raise AttributeError("Dataset item missing attribute 'outpin_nids'.")
        nids = getattr(it, "outpin_nids")
        if nids is None:
            raise RuntimeError("outpin_nids is None. Your dataset must provide outpin_nids for distillation.")
        if not torch.is_tensor(nids):
            nids = torch.as_tensor(nids, dtype=torch.int64)
        nids = nids.to(torch.int64).view(-1)
        if nids.numel() > 0:
            nids_all.append(nids + int(offset))
        offset += _num_target_nodes(it.g_cone, hetero=hetero, target_ntype=target_ntype)
    if len(nids_all) == 0:
        return torch.zeros((0,), dtype=torch.int64)
    return torch.cat(nids_all, dim=0)


def _batch_center_nids_3hop(batch: List[Any], *, hetero: bool, target_ntype: str = "pin") -> torch.Tensor:
    """Build batched center node ids for a list of 3-hop graphs."""
    nids_all: List[torch.Tensor] = []
    offset = 0
    for g, center_nid in batch:
        center = torch.as_tensor([int(center_nid)], dtype=torch.int64)
        nids_all.append(center + int(offset))
        offset += _num_target_nodes(g, hetero=hetero, target_ntype=target_ntype)
    if len(nids_all) == 0:
        return torch.zeros((0,), dtype=torch.int64)
    return torch.cat(nids_all, dim=0)


def _center_nid_from_payload(payload: Dict[str, Any]) -> int:
    if "node_names" not in payload:
        raise RuntimeError("3-hop chunk payload missing node_names; re-export with write_node_names=True.")
    center_name = str(payload["storage_key"]["center_name"])
    try:
        return int(list(payload["node_names"]).index(center_name))
    except ValueError as e:
        raise RuntimeError(f"center_name not found in payload node_names: {center_name}") from e


def _mask_for_node_labels(y: torch.Tensor) -> torch.Tensor:
    """Regression supervision mask: valid where finite and not sentinel -1."""
    if y.dim() == 1:
        return torch.isfinite(y) & (y != -1.0)
    return torch.isfinite(y).all(dim=1) & (y[:, 0] != -1.0)


def _eval_reg_mae(pred: torch.Tensor, y: torch.Tensor) -> float:
    valid = torch.isfinite(y).all(dim=1) & (y[:, 0] != -1.0)
    if not valid.any():
        return 1e9
    return float((pred[valid] - y[valid]).abs().mean().item())


def _batch_regression_stats_1d(y_pred_1d: torch.Tensor, y_true_1d: torch.Tensor) -> Dict[str, float]:
    """Compute MAE/MSE/R2 for 1D regression on the current batch."""
    yt = y_true_1d.detach()
    yp = y_pred_1d.detach()
    fin = torch.isfinite(yp) & torch.isfinite(yt)
    if not fin.any():
        return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan")}
    yt = yt[fin]
    yp = yp[fin]
    diff = yp - yt
    mae = float(diff.abs().mean().item())
    mse = float((diff * diff).mean().item())
    y_mean = yt.mean()
    sse = float((diff * diff).sum().item())
    sst = float(((yt - y_mean) * (yt - y_mean)).sum().item())
    r2 = float("nan") if sst <= 1e-12 else float(1.0 - sse / sst)
    return {"mae": mae, "mse": mse, "r2": r2}


@torch.no_grad()
def _sanitize_model(model: torch.nn.Module) -> int:
    """Replace NaN/Inf in parameters and buffers. Returns count of fixed tensors."""
    fixed = 0
    for p in model.parameters():
        if not torch.isfinite(p.data).all():
            p.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            fixed += 1
    for _name, buf in model.named_buffers():
        if buf is not None and buf.is_floating_point() and not torch.isfinite(buf).all():
            buf.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            fixed += 1
    return fixed


class _RunningRegStats:
    """Streaming stats for epoch-level MAE/MSE/R2 (1D or flattened multi-d)."""

    def __init__(self) -> None:
        self.abs_err_sum = 0.0
        self.sse = 0.0
        self.n = 0
        self._count = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        y_cpu = y_true.detach().to("cpu", dtype=torch.float64).view(-1)
        p_cpu = y_pred.detach().to("cpu", dtype=torch.float64).view(-1)
        valid = torch.isfinite(p_cpu) & torch.isfinite(y_cpu)
        if not valid.any():
            return
        y_cpu = y_cpu[valid]
        p_cpu = p_cpu[valid]
        diff = p_cpu - y_cpu
        self.abs_err_sum += float(diff.abs().sum().item())
        self.sse += float((diff * diff).sum().item())
        self.n += int(y_cpu.numel())
        n_b = int(y_cpu.numel())
        if n_b == 0:
            return
        mean_b = float(y_cpu.mean().item())
        M2_b = float(((y_cpu - mean_b) ** 2).sum().item())
        n_a = self._count
        self._count = n_a + n_b
        delta = mean_b - self._mean
        self._mean += delta * n_b / self._count
        self._M2 += M2_b + delta * delta * n_a * n_b / self._count

    def finalize(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan")}
        mae = self.abs_err_sum / max(1, self.n)
        mse = self.sse / max(1, self.n)
        r2 = float("nan") if self._M2 <= 1e-12 else float(1.0 - (self.sse / self._M2))
        return {"mae": float(mae), "mse": float(mse), "r2": float(r2)}


def _infer_out_dim_from_y(*, y_example: torch.Tensor) -> int:
    """Infer regression output dimension from an example label tensor."""
    if y_example.dim() == 1:
        return 1
    return int(y_example.shape[1])


def _save_ckpt(path: Path, *, model: torch.nn.Module, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, str(path))


def load_student_from_ckpt(
    y_key: str,
    *,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
) -> Student3HopEncoder:
    """Instantiate and load a Student3HopEncoder from a checkpoint.

    Args:
        y_key: Target key used during training (e.g. ``"slack_eco"``).
        ckpt_path: Optional explicit path to the ``.pt`` checkpoint.
        device: torch device string.

    Returns:
        Student3HopEncoder in eval mode, weights loaded, moved to *device*.
    """
    path = ckpt_path if ckpt_path is not None else str(student_ckpt_path(y_key))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}

    student = Student3HopEncoder(
        hetero=bool(cfg["hetero"]),
        gnn_type=str(cfg.get("gnn_type_student", cfg.get("gnn_type", "gat"))),
        x_keys=list(cfg["x_keys"]),
        maxType=int(cfg["maxType"]),
        max_size=int(cfg["max_size"]),
        hid_dim=int(cfg["hid_dim"]),
        emb_dim=int(cfg["emb_dim"]),
        out_dim=int(cfg["out_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
    )

    sd = ck["state_dict"]
    if any(k.startswith("local_enc.") for k in sd.keys()):
        w = sd.get("local_enc.proj.weight", None)
        if torch.is_tensor(w) and w.dim() == 2:
            student._lazy_build_local(int(w.shape[1]))

    student.load_state_dict(sd, strict=True)
    student.to(torch.device(device))
    student.eval()
    return student


def load_teacher_from_ckpt(
    y_key: str,
    *,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
) -> TeacherConeSGFormer:
    """Instantiate and load a TeacherConeSGFormer from a checkpoint.

    Args:
        y_key: Target key used during training (e.g. ``"slack_eco"``).
        ckpt_path: Optional explicit path to the ``.pt`` checkpoint.
        device: torch device string.

    Returns:
        TeacherConeSGFormer in eval mode, weights loaded, moved to *device*.
    """
    path = ckpt_path if ckpt_path is not None else str(teacher_ckpt_path(y_key))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}

    teacher = TeacherConeSGFormer(
        hetero=bool(cfg["hetero"]),
        x_keys=list(cfg["x_keys"]),
        maxType=int(cfg["maxType"]),
        max_size=int(cfg["max_size"]),
        hid_dim=int(cfg["hid_dim"]),
        emb_dim=int(cfg["emb_dim"]),
        out_dim=int(cfg["out_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
        local_gnn=str(cfg.get("teacher_local_gnn", "gcn")),
        trans_num_layers=int(cfg.get("teacher_trans_layers", 1)),
        trans_num_heads=int(cfg.get("teacher_trans_heads", 1)),
    )

    sd = ck["state_dict"]
    w = sd.get("global_enc.in_fc.weight", None)
    if w is None:
        w = sd.get("local_enc.proj.weight", None)
    if w is None:
        raise RuntimeError(
            "Cannot infer in_dim for TeacherConeSGFormer from state_dict. "
            "Expected key 'global_enc.in_fc.weight' or 'local_enc.proj.weight'."
        )
    in_dim = int(w.shape[1])
    teacher._lazy_build(in_dim)

    teacher.load_state_dict(sd, strict=True)
    teacher.to(torch.device(device))
    teacher.eval()
    return teacher


@torch.no_grad()
def build_retrieval_index(
    *,
    y_key: str,
    chunk_dir: str = chunk_dir,
    student_ckpt: Optional[str] = None,
    out_subdir: str,
    index_split: str = "train",
    device: str,
    hetero: bool,
    gnn_type: str,
    x_keys: List[str],
    maxType: int,
    max_size: int,
    hid_dim: int,
    emb_dim: int,
    dropout: float,
    norm_stats_3hop: Dict[str, Any],
    signatures: Optional[List[str]] = None,
    batch_size: int = 128,
) -> None:
    """Encode 3-hop chunks with the student encoder and save the retrieval index.

    Uses student mean-pooled embedding as retrieval key and the student
    regression prediction as retrieval value.
    """
    dev = torch.device(device)

    if student_ckpt is None:
        student_ckpt = str(student_ckpt_path(y_key))

    ck = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {}) or {}
    task = str(cfg.get("task", _infer_task(y_key)))
    if "out_dim" not in cfg:
        raise RuntimeError(f"student_ckpt cfg missing out_dim: {student_ckpt}")
    out_dim = int(cfg["out_dim"])

    hetero_i = bool(cfg.get("hetero", hetero))
    gnn_type_i = str(cfg.get("gnn_type_student", cfg.get("gnn_type", gnn_type)))
    x_keys_i = list(cfg.get("x_keys", x_keys))
    maxType_i = int(cfg.get("maxType", maxType))
    max_size_i = int(cfg.get("max_size", max_size))
    hid_dim_i = int(cfg.get("hid_dim", hid_dim))
    emb_dim_i = int(cfg.get("emb_dim", emb_dim))
    dropout_i = float(cfg.get("dropout", dropout))

    student = Student3HopEncoder(
        hetero=hetero_i,
        gnn_type=gnn_type_i,
        x_keys=x_keys_i,
        maxType=maxType_i,
        max_size=max_size_i,
        hid_dim=hid_dim_i,
        emb_dim=emb_dim_i,
        out_dim=out_dim,
        dropout=dropout_i,
    ).to(dev)

    sd: Dict[str, torch.Tensor] = ck["state_dict"]

    def _infer_local_in_dim_from_sd(state_dict: Dict[str, torch.Tensor]) -> int:
        w = state_dict.get("local_enc.proj.weight", None)
        if torch.is_tensor(w) and w.dim() == 2:
            return int(w.shape[1])
        candidates: List[Tuple[str, torch.Tensor]] = []
        for k, v in state_dict.items():
            if not k.startswith("local_enc."):
                continue
            if not torch.is_tensor(v) or v.dim() != 2:
                continue
            candidates.append((k, v))
        if not candidates:
            raise RuntimeError(
                "Cannot infer local_enc input dim from state_dict. "
                "Expected keys like 'local_enc.proj.weight' or other 2D weights under 'local_enc.*'."
            )
        candidates.sort(key=lambda kv: kv[0])
        return int(candidates[0][1].shape[1])

    if any(k.startswith("local_enc.") for k in sd.keys()):
        in_dim = _infer_local_in_dim_from_sd(sd)
        student._lazy_build_local(int(in_dim))

    student.load_state_dict(sd, strict=True)
    student.eval()

    layout = ChunkLayout(chunk_dir)
    idx_path = os.path.join(chunk_dir, "index.jsonl")
    rows: List[Dict[str, Any]] = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if signatures is None:
        sigs = [r["signature"] for r in rows if "signature" in r]
        metas = [r.get("meta", {}) or {} for r in rows if "signature" in r]
    else:
        sig_set = set(signatures)
        sigs, metas = [], []
        for r in rows:
            s = r.get("signature", None)
            if s is None:
                continue
            if s in sig_set:
                sigs.append(s)
                metas.append(r.get("meta", {}) or {})

    out_subdir_task = f"{out_subdir}_{task}"
    out_root = Path(chunk_dir) / "retrieval_index" / out_subdir_task / index_split / y_key
    out_root.mkdir(parents=True, exist_ok=True)
    out_emb = out_root / "embeddings.pt"
    out_val = out_root / "values.pt"
    out_map = out_root / "mapping.jsonl"

    import dgl

    all_emb: List[torch.Tensor] = []
    all_val: List[torch.Tensor] = []
    mapping: List[Dict[str, Any]] = []

    for i in tqdm.tqdm(range(0, len(sigs), batch_size), desc="Encoding chunks", colour="cyan"):
        batch_sigs = sigs[i : i + batch_size]
        g_list = []
        payload_list: List[Dict[str, Any]] = []
        for s in batch_sigs:
            payload = torch.load(layout.chunk_path(s), map_location="cpu", weights_only=False)
            payload_list.append(payload)
            g = chunk_to_dgl(
                payload,
                hetero=hetero_i,
                build_undirected_hops=True,
                device=dev,
                normalize=True,
                norm_stats=norm_stats_3hop,
            )
            g_list.append(g)

        bg = dgl.batch(g_list).to(dev)
        yhat, z = student(bg)
        z_cpu = z.detach().float().cpu()
        y_cpu = yhat.detach().float().cpu()
        all_emb.append(z_cpu)
        all_val.append(y_cpu)

        for j, s in enumerate(batch_sigs):
            meta = metas[i + j] if (i + j) < len(metas) else {}
            mapping.append(
                {
                    "row": i + j,
                    "signature": s,
                    "design_id": meta.get("design_id"),
                    "center_name": meta.get("center_name"),
                    "center_type": meta.get("center_type"),
                }
            )

    emb = (
        torch.cat(all_emb, dim=0)
        if len(all_emb) > 0
        else torch.zeros((0, emb_dim_i), dtype=torch.float32)
    )
    val = (
        torch.cat(all_val, dim=0)
        if len(all_val) > 0
        else torch.zeros((0, out_dim), dtype=torch.float32)
    )
    torch.save({"task": task, "y_key": y_key, "signatures": sigs, "embeddings": emb, "index_split": index_split}, out_emb)
    torch.save({"task": task, "y_key": y_key, "signatures": sigs, "values": val, "index_split": index_split}, out_val)

    with open(out_map, "w", encoding="utf-8") as f:
        for row in mapping:
            f.write(json.dumps(row) + "\n")

    print(f"[Index][{y_key}] task={task} split={index_split} encoded {len(sigs)} chunks -> {out_root}")


def build_norm_stats(
    train_designs: List[str],
    *,
    ep_dir: str = ep_dir,
    chunk_dir: str = chunk_dir,
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute train-only normalisation statistics for cone and 3-hop datasets.

    Returns:
        ``(norm_stats_cone, norm_stats_3hop)``
    """
    cone_train_sigs = signatures_by_designs(train_designs, out_dir=ep_dir)
    hop_train_sigs = signatures_by_designs(train_designs, out_dir=chunk_dir)

    if len(cone_train_sigs) == 0:
        raise RuntimeError(
            "No logic-cone signatures found for train_designs in ep_dir/index.jsonl meta.design_id."
        )
    if len(hop_train_sigs) == 0:
        raise RuntimeError(
            "No 3-hop signatures found for train_designs in chunk_dir/index.jsonl meta.design_id."
        )

    norm_stats_cone = build_train_norm_stats(
        cone_train_sigs,
        out_dir=ep_dir,
        fields_node=norm_fields_node,
        fields_edge=norm_fields_edge,
        designs=train_designs,
    )
    norm_stats_3hop = build_train_norm_stats(
        hop_train_sigs,
        out_dir=chunk_dir,
        fields_node=norm_fields_node,
        fields_edge=norm_fields_edge,
        designs=train_designs,
    )
    return norm_stats_cone, norm_stats_3hop


def train_one_task(
    *,
    y_key: str,
    ep_dir: str = ep_dir,
    chunk_dir: str = chunk_dir,
    checkpoint_dir: str = out_dir,
    device: str,
    hetero: bool,
    gnn_type_student: str,
    x_keys: List[str],
    maxType: int,
    max_size: int,
    hid_dim: int,
    emb_dim: int,
    dropout: float,
    teacher_local_gnn: str,
    teacher_trans_layers: int,
    teacher_trans_heads: int,
    epochs: int,
    batch_size_cone: int,
    lr_teacher: float,
    lr_student: float,
    lambda_distill: float,
    train_designs: List[str],
    test_designs: List[str],
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
    norm_stats_cone: Optional[Dict[str, Any]] = None,
    norm_stats_3hop: Optional[Dict[str, Any]] = None,
    log_every: int = 0,
    label_norm: bool = False,
    label_norm_ref_key: Optional[str] = "slack",
) -> Dict[str, Any]:
    """Train teacher on logic cones and distill student on 3-hop chunks.

    Only regression targets (e.g. ``slack_eco``, ``slack``) are supported.

    Best checkpoints are saved to ``student_ckpt_path(y_key)`` and
    ``teacher_ckpt_path(y_key)``.  Use :func:`load_student_from_ckpt` /
    :func:`load_teacher_from_ckpt` to reload them by *y_key* after training.

    *norm_stats_cone* and *norm_stats_3hop* can be pre-computed via
    :func:`build_norm_stats` and passed in directly to skip recomputation.
    """
    import dgl

    dev = torch.device(device)
    task = _infer_task(y_key)  # always "reg"

    # ----------------------------
    # split by designs
    # ----------------------------
    cone_train_sigs = signatures_by_designs(train_designs, out_dir=ep_dir)
    cone_test_sigs = signatures_by_designs(test_designs, out_dir=ep_dir)
    hop_train_sigs = signatures_by_designs(train_designs, out_dir=chunk_dir)
    hop_test_sigs = signatures_by_designs(test_designs, out_dir=chunk_dir)

    if len(cone_train_sigs) == 0:
        raise RuntimeError("No logic-cone signatures found for train_designs in ep_dir/index.jsonl meta.design_id.")
    if len(cone_test_sigs) == 0:
        raise RuntimeError("No logic-cone signatures found for test_designs in ep_dir/index.jsonl meta.design_id.")
    if len(hop_train_sigs) == 0:
        raise RuntimeError("No 3-hop signatures found for train_designs in chunk_dir/index.jsonl meta.design_id.")
    if len(hop_test_sigs) == 0:
        raise RuntimeError("No 3-hop signatures found for test_designs in chunk_dir/index.jsonl meta.design_id.")

    # ----------------------------
    # train-only norm stats
    # ----------------------------
    if norm_stats_cone is None or norm_stats_3hop is None:
        _cone, _hop = build_norm_stats(
            train_designs,
            ep_dir=ep_dir,
            chunk_dir=chunk_dir,
            norm_fields_node=norm_fields_node,
            norm_fields_edge=norm_fields_edge,
        )
        if norm_stats_cone is None:
            norm_stats_cone = _cone
        if norm_stats_3hop is None:
            norm_stats_3hop = _hop

    # ----------------------------
    # per-design label normalization stats
    # ----------------------------
    from utils.chunk_graph_data import compute_per_design_label_stats as _compute_lns
    lns: Optional[Dict[str, Any]] = None
    if label_norm:
        all_designs = list(set(list(train_designs) + list(test_designs)))
        lns = _compute_lns(
            all_designs, out_dir=ep_dir, y_feat_key=y_key,
            label_norm_ref_key=label_norm_ref_key,
        )
        print(f"[distill][label_norm] Per-design label stats (ref={label_norm_ref_key}):")
        for did, s in lns.items():
            print(f"  {did}: mean={s['mean']:.6f} std={s['std']:.6f} n={s['n']}")

    # ----------------------------
    # datasets & loaders
    # ----------------------------
    train_ds = Cone2Outpin3HopDataset(
        ep_dir=ep_dir,
        chunk_dir=chunk_dir,
        cone_signatures=cone_train_sigs,
        hetero=hetero,
        build_undirected_hops=True,
        device=dev,
        normalize=True,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        norm_stats_cone=norm_stats_cone,
        norm_stats_3hop=norm_stats_3hop,
        y_key=y_key,
    )
    test_ds = Cone2Outpin3HopDataset(
        ep_dir=ep_dir,
        chunk_dir=chunk_dir,
        cone_signatures=cone_test_sigs,
        hetero=hetero,
        build_undirected_hops=True,
        device=dev,
        normalize=True,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        norm_stats_cone=norm_stats_cone,
        norm_stats_3hop=norm_stats_3hop,
        y_key=y_key,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size_cone, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    test_dl = DataLoader(test_ds, batch_size=batch_size_cone, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    # ----------------------------
    # Infer out_dim from first train batch labels
    # ----------------------------
    first_batch = next(iter(train_dl))
    y_out_list_ex: List[torch.Tensor] = []
    for it in first_batch:
        y_out_list_ex.extend(it.y_outpin_list)
    if len(y_out_list_ex) == 0:
        raise RuntimeError(f"[{y_key}] Cannot infer out_dim: first batch has empty y_outpin_list.")

    y_ex = torch.stack([t if t.dim() > 0 else t.view(1) for t in y_out_list_ex], dim=0).to("cpu")
    out_dim = _infer_out_dim_from_y(y_example=y_ex)

    # ----------------------------
    # models / optimizers
    # ----------------------------
    teacher = TeacherConeSGFormer(
        hetero=hetero,
        x_keys=x_keys,
        maxType=maxType,
        max_size=max_size,
        hid_dim=hid_dim,
        emb_dim=emb_dim,
        out_dim=out_dim,
        dropout=dropout,
        local_gnn=teacher_local_gnn,
        trans_num_layers=teacher_trans_layers,
        trans_num_heads=teacher_trans_heads,
    ).to(dev)

    student = Student3HopEncoder(
        hetero=hetero,
        gnn_type=gnn_type_student,
        x_keys=x_keys,
        maxType=maxType,
        max_size=max_size,
        hid_dim=hid_dim,
        emb_dim=emb_dim,
        out_dim=out_dim,
        dropout=dropout,
    ).to(dev)

    opt_t = torch.optim.AdamW(teacher.parameters(), lr=lr_teacher, weight_decay=1e-4)
    opt_s = torch.optim.AdamW(student.parameters(), lr=lr_student, weight_decay=1e-4)
    sched_t = CosineAnnealingLR(opt_t, T_max=epochs, eta_min=1e-6)
    sched_s = CosineAnnealingLR(opt_s, T_max=epochs, eta_min=1e-6)
    global_step = 0

    best_student_path = student_ckpt_path(y_key, checkpoint_dir=checkpoint_dir)
    best_teacher_path = teacher_ckpt_path(y_key, checkpoint_dir=checkpoint_dir)
    if best_student_path.exists():
        best_student_path.unlink()
    if best_teacher_path.exists():
        best_teacher_path.unlink()
    best_metric = 1e18  # minimize MAE

    nan_streak_t = 0
    nan_streak_s = 0

    for ep in range(epochs):
        teacher.train()
        student.train()

        tr_t_loss_sum = 0.0
        tr_t_n = 0
        tr_s_loss_sum = 0.0
        tr_s_n = 0
        tr_t_reg = _RunningRegStats()
        tr_s_reg = _RunningRegStats()

        for bidx, batch in enumerate(train_dl, start=1):
            global_step += 1

            # ---- teacher: cone node-level supervised ----
            g_cone = dgl.batch([it.g_cone for it in batch]).to(dev)
            y_cone = torch.cat([it.y_cone_all for it in batch], dim=0).to(dev)

            _lm_cone = _ls_cone = None
            if lns is not None:
                _lm_parts, _ls_parts = [], []
                for it in batch:
                    did = it.cone_meta.get("design_id", "")
                    s = lns.get(did, {"mean": 0.0, "std": 1.0})
                    nn = int(it.y_cone_all.shape[0])
                    _lm_parts.append(torch.full((nn,), s["mean"], dtype=torch.float32))
                    _ls_parts.append(torch.full((nn,), s["std"], dtype=torch.float32))
                _lm_cone = torch.cat(_lm_parts).to(dev)
                _ls_cone = torch.cat(_ls_parts).to(dev)
                y_cone = (y_cone.float() - _lm_cone.unsqueeze(-1) if y_cone.dim() == 2 else y_cone.float() - _lm_cone) / (
                    _ls_cone.unsqueeze(-1) if y_cone.dim() == 2 else _ls_cone
                )

            outpin_nids = _batch_outpin_nids(batch, hetero=hetero, target_ntype="pin").to(dev)
            node_yhat_t, _node_z_t, outpin_z_t, _h_raw_t = teacher(g_cone, outpin_nids=outpin_nids)

            m = _mask_for_node_labels(y_cone)
            loss_t = F.mse_loss(node_yhat_t[m], y_cone[m]) if m.any() else (node_yhat_t.sum() * 0.0)

            with torch.no_grad():
                if m.any():
                    yhat_denorm = node_yhat_t[m]
                    y_denorm = y_cone[m]
                    if _lm_cone is not None:
                        _lm_m = _lm_cone[m]
                        _ls_m = _ls_cone[m]
                        if yhat_denorm.dim() == 2:
                            yhat_denorm = yhat_denorm * _ls_m.unsqueeze(1) + _lm_m.unsqueeze(1)
                            y_denorm = y_denorm * _ls_m.unsqueeze(1) + _lm_m.unsqueeze(1)
                        else:
                            yhat_denorm = yhat_denorm * _ls_m + _lm_m
                            y_denorm = y_denorm * _ls_m + _lm_m
                    tr_t_reg.update(yhat_denorm, y_denorm)

                    if log_every > 0 and (bidx % log_every == 0):
                        if y_cone.dim() == 1 or (y_cone.dim() == 2 and y_cone.shape[1] == 1):
                            yt_b = y_denorm.view(-1)
                            yp_b = yhat_denorm.view(-1)
                            stats_b = _batch_regression_stats_1d(yp_b, yt_b)
                            print(
                                f"[Ep {ep:03d} | step {global_step:06d} | batch {bidx:04d}] "
                                f"teacher_loss={float(loss_t.item()):.6f} "
                                f"mae={stats_b['mae']:.6f} mse={stats_b['mse']:.6f} r2={stats_b['r2']:.6f} "
                                f"valid_n={int(m.sum().item())}"
                            )
                        else:
                            print(
                                f"[Ep {ep:03d} | step {global_step:06d} | batch {bidx:04d}] "
                                f"teacher_loss={float(loss_t.item()):.6f} "
                                f"valid_n={int(m.sum().item())}"
                            )

            _lt = float(loss_t.detach().item())
            if math.isfinite(_lt):
                tr_t_loss_sum += _lt * int(m.sum().item() if m.any() else 0)
                tr_t_n += int(m.sum().item()) if m.any() else 0

            if torch.isfinite(loss_t):
                nan_streak_t = 0
                opt_t.zero_grad(set_to_none=True)
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), 5.0)
                opt_t.step()
                nf = _sanitize_model(teacher)
                if nf > 0:
                    logger.warning(f"[Ep {ep} batch {bidx}] fixed {nf} NaN tensors in teacher after step")
            else:
                nan_streak_t += 1
                _sanitize_model(teacher)
                logger.warning(f"[Ep {ep} batch {bidx}] teacher loss is NaN/Inf, skipping step (streak={nan_streak_t})")

            # ---- student: distill on 3-hop chunks ----
            g3_list = []
            y_out_list = []
            y_out_dids: List[str] = []
            for it in batch:
                g3_list.extend(it.g_3hop_list)
                y_out_list.extend(it.y_outpin_list)
                did = it.cone_meta.get("design_id", "")
                y_out_dids.extend([did] * len(it.y_outpin_list))
            if len(g3_list) == 0:
                continue

            bg3 = dgl.batch(g3_list).to(dev)
            y_out = torch.stack([t if t.dim() > 0 else t.view(1) for t in y_out_list], dim=0).to(dev)

            _lm_s = _ls_s = None
            if lns is not None:
                _s_means = [lns.get(d, {"mean": 0.0, "std": 1.0})["mean"] for d in y_out_dids]
                _s_stds = [lns.get(d, {"mean": 0.0, "std": 1.0})["std"] for d in y_out_dids]
                _lm_s = torch.tensor(_s_means, dtype=torch.float32, device=dev)
                _ls_s = torch.tensor(_s_stds, dtype=torch.float32, device=dev)
                if y_out.dim() == 2:
                    y_out = (y_out.float() - _lm_s.unsqueeze(1)) / _ls_s.unsqueeze(1)
                else:
                    y_out = (y_out.float() - _lm_s) / _ls_s

            yhat_s, outpin_z_s = student(bg3)

            valid = torch.isfinite(y_out).all(dim=1) & (y_out[:, 0] != -1.0)
            loss_sup = F.mse_loss(yhat_s[valid], y_out[valid]) if valid.any() else (yhat_s.sum() * 0.0)

            if outpin_z_t.shape[0] != outpin_z_s.shape[0]:
                raise RuntimeError(
                    f"Distill shape mismatch: teacher outpin_z {tuple(outpin_z_t.shape)} vs "
                    f"student outpin_z {tuple(outpin_z_s.shape)}. "
                    "Check that outpin_nids is aligned with g_3hop_list order."
                )
            loss_distill = F.mse_loss(outpin_z_s, outpin_z_t.detach())
            loss_s = loss_sup + lambda_distill * loss_distill if torch.isfinite(loss_distill) else loss_sup

            with torch.no_grad():
                valid_s = torch.isfinite(y_out).all(dim=1) & (y_out[:, 0] != -1.0)
                n_valid_s = int(valid_s.sum().item())
                if n_valid_s > 0:
                    yhat_s_denorm = yhat_s[valid_s]
                    y_out_denorm = y_out[valid_s]
                    if _lm_s is not None:
                        if yhat_s_denorm.dim() == 2:
                            yhat_s_denorm = yhat_s_denorm * _ls_s[valid_s].unsqueeze(1) + _lm_s[valid_s].unsqueeze(1)
                            y_out_denorm = y_out_denorm * _ls_s[valid_s].unsqueeze(1) + _lm_s[valid_s].unsqueeze(1)
                        else:
                            yhat_s_denorm = yhat_s_denorm * _ls_s[valid_s] + _lm_s[valid_s]
                            y_out_denorm = y_out_denorm * _ls_s[valid_s] + _lm_s[valid_s]
                    tr_s_reg.update(yhat_s_denorm, y_out_denorm)

                    if log_every > 0 and (bidx % log_every == 0):
                        if y_out.shape[1] == 1:
                            stats_b = _batch_regression_stats_1d(yhat_s_denorm.view(-1), y_out_denorm.view(-1))
                            print(
                                f"[Ep {ep:03d} | step {global_step:06d} | batch {bidx:04d}] "
                                f"student_loss={float(loss_s.item()):.6f} "
                                f"(sup={float(loss_sup.item()):.6f}, distill={float(loss_distill.item()):.6f}) "
                                f"mae={stats_b['mae']:.6f} mse={stats_b['mse']:.6f} r2={stats_b['r2']:.6f} "
                                f"valid_n={n_valid_s}"
                            )
                        else:
                            print(
                                f"[Ep {ep:03d} | step {global_step:06d} | batch {bidx:04d}] "
                                f"student_loss={float(loss_s.item()):.6f} "
                                f"(sup={float(loss_sup.item()):.6f}, distill={float(loss_distill.item()):.6f}) "
                                f"valid_n={n_valid_s}"
                            )

                    _ls = float(loss_s.detach().item())
                    if math.isfinite(_ls):
                        tr_s_loss_sum += _ls * n_valid_s
                        tr_s_n += n_valid_s

            if torch.isfinite(loss_s):
                nan_streak_s = 0
                opt_s.zero_grad(set_to_none=True)
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                opt_s.step()
                nf = _sanitize_model(student)
                if nf > 0:
                    logger.warning(f"[Ep {ep} batch {bidx}] fixed {nf} NaN tensors in student after step")
            else:
                nan_streak_s += 1
                _sanitize_model(student)
                logger.warning(f"[Ep {ep} batch {bidx}] student loss is NaN/Inf, skipping step (streak={nan_streak_s})")

        # ----------------------------
        # Epoch summary (train)
        # ----------------------------
        tr_t_loss = tr_t_loss_sum / max(1, tr_t_n)
        tr_s_loss = tr_s_loss_sum / max(1, tr_s_n)
        tr_t_reg_s = tr_t_reg.finalize()
        tr_s_reg_s = tr_s_reg.finalize()

        # ----------------------------
        # Eval on test split
        # ----------------------------
        teacher.eval()
        student.eval()

        te_t_loss_sum = 0.0
        te_t_n = 0
        te_t_reg = _RunningRegStats()
        te_s_reg = _RunningRegStats()
        te_has_student = False

        with torch.no_grad():
            for batch in test_dl:
                g_cone = dgl.batch([it.g_cone for it in batch]).to(dev)
                y_cone_raw = torch.cat([it.y_cone_all for it in batch], dim=0).to(dev)
                outpin_nids = _batch_outpin_nids(batch, hetero=hetero, target_ntype="pin").to(dev)

                y_cone = y_cone_raw
                _lm_cone_te = _ls_cone_te = None
                if lns is not None:
                    _lm_parts, _ls_parts = [], []
                    for it in batch:
                        did = it.cone_meta.get("design_id", "")
                        s = lns.get(did, {"mean": 0.0, "std": 1.0})
                        nn = int(it.y_cone_all.shape[0])
                        _lm_parts.append(torch.full((nn,), s["mean"], dtype=torch.float32))
                        _ls_parts.append(torch.full((nn,), s["std"], dtype=torch.float32))
                    _lm_cone_te = torch.cat(_lm_parts).to(dev)
                    _ls_cone_te = torch.cat(_ls_parts).to(dev)
                    if y_cone.dim() == 2:
                        y_cone = (y_cone.float() - _lm_cone_te.unsqueeze(1)) / _ls_cone_te.unsqueeze(1)
                    else:
                        y_cone = (y_cone.float() - _lm_cone_te) / _ls_cone_te

                node_yhat_t, _node_z_t, _outpin_z_t, _h_raw_t = teacher(g_cone, outpin_nids=outpin_nids)

                m = _mask_for_node_labels(y_cone)
                if m.any():
                    loss = F.mse_loss(node_yhat_t[m], y_cone[m])
                    yhat_denorm = node_yhat_t[m]
                    y_denorm = y_cone[m]
                    if _lm_cone_te is not None:
                        _lm_m = _lm_cone_te[m]
                        _ls_m = _ls_cone_te[m]
                        if yhat_denorm.dim() == 2:
                            yhat_denorm = yhat_denorm * _ls_m.unsqueeze(1) + _lm_m.unsqueeze(1)
                            y_denorm = y_denorm * _ls_m.unsqueeze(1) + _lm_m.unsqueeze(1)
                        else:
                            yhat_denorm = yhat_denorm * _ls_m + _lm_m
                            y_denorm = y_denorm * _ls_m + _lm_m
                    te_t_reg.update(yhat_denorm, y_denorm)
                    n_valid = int(m.sum().item())
                    te_t_loss_sum += float(loss.item()) * n_valid
                    te_t_n += n_valid

                # ---- student (outpin-level) ----
                g3_list = []
                y_out_list = []
                y_out_dids_te: List[str] = []
                for it in batch:
                    g3_list.extend(it.g_3hop_list)
                    y_out_list.extend(it.y_outpin_list)
                    did = it.cone_meta.get("design_id", "")
                    y_out_dids_te.extend([did] * len(it.y_outpin_list))
                if len(g3_list) == 0:
                    continue

                bg3 = dgl.batch(g3_list).to(dev)
                y_out = torch.stack([t if t.dim() > 0 else t.view(1) for t in y_out_list], dim=0).to(dev)

                _lm_s_te = _ls_s_te = None
                if lns is not None:
                    _s_m = [lns.get(d, {"mean": 0.0, "std": 1.0})["mean"] for d in y_out_dids_te]
                    _s_s = [lns.get(d, {"mean": 0.0, "std": 1.0})["std"] for d in y_out_dids_te]
                    _lm_s_te = torch.tensor(_s_m, dtype=torch.float32, device=dev)
                    _ls_s_te = torch.tensor(_s_s, dtype=torch.float32, device=dev)
                    if y_out.dim() == 2:
                        y_out = (y_out.float() - _lm_s_te.unsqueeze(1)) / _ls_s_te.unsqueeze(1)
                    else:
                        y_out = (y_out.float() - _lm_s_te) / _ls_s_te

                yhat_s, _outpin_z_s = student(bg3)

                valid_s = torch.isfinite(y_out).all(dim=1) & (y_out[:, 0] != -1.0)
                if valid_s.any():
                    yhat_s_denorm = yhat_s[valid_s]
                    y_out_denorm = y_out[valid_s]
                    if _lm_s_te is not None:
                        if yhat_s_denorm.dim() == 2:
                            yhat_s_denorm = yhat_s_denorm * _ls_s_te[valid_s].unsqueeze(1) + _lm_s_te[valid_s].unsqueeze(1)
                            y_out_denorm = y_out_denorm * _ls_s_te[valid_s].unsqueeze(1) + _lm_s_te[valid_s].unsqueeze(1)
                        else:
                            yhat_s_denorm = yhat_s_denorm * _ls_s_te[valid_s] + _lm_s_te[valid_s]
                            y_out_denorm = y_out_denorm * _ls_s_te[valid_s] + _lm_s_te[valid_s]
                    te_s_reg.update(yhat_s_denorm, y_out_denorm)
                    te_has_student = True

        sched_t.step()
        sched_s.step()

        if not te_has_student:
            continue

        # ---- compute metrics and checkpoint ----
        _student_cfg = dict(
            y_key=y_key, task=task, out_dim=out_dim, emb_dim=emb_dim,
            hetero=hetero, gnn_type_student=gnn_type_student,
            x_keys=x_keys, maxType=int(maxType), max_size=int(max_size),
            hid_dim=int(hid_dim), dropout=float(dropout),
        )
        _teacher_cfg = dict(
            y_key=y_key, task=task, out_dim=out_dim, emb_dim=emb_dim,
            hetero=hetero, x_keys=x_keys, maxType=int(maxType),
            max_size=int(max_size), hid_dim=int(hid_dim), dropout=float(dropout),
            teacher_local_gnn=teacher_local_gnn,
            teacher_trans_layers=int(teacher_trans_layers),
            teacher_trans_heads=int(teacher_trans_heads),
        )

        te_t_loss = te_t_loss_sum / max(1, te_t_n)
        te_t_reg_s = te_t_reg.finalize()
        te_s_reg_s = te_s_reg.finalize()

        metric = float(te_s_reg_s["mae"])
        improved = metric < best_metric
        if improved:
            best_metric = metric
            _save_ckpt(best_student_path, model=student, cfg=_student_cfg)
            _save_ckpt(best_teacher_path, model=teacher, cfg=_teacher_cfg)

        print(
            f"[{y_key}][Ep {ep:03d}] "
            f"T(tr) loss={tr_t_loss:.6f} mae={tr_t_reg_s['mae']:.6f} mse={tr_t_reg_s['mse']:.6f} r2={tr_t_reg_s['r2']:.6f} | "
            f"T(te) loss={te_t_loss:.6f} mae={te_t_reg_s['mae']:.6f} mse={te_t_reg_s['mse']:.6f} r2={te_t_reg_s['r2']:.6f} | "
            f"S(tr) loss={tr_s_loss:.6f} mae={tr_s_reg_s['mae']:.6f} mse={tr_s_reg_s['mse']:.6f} r2={tr_s_reg_s['r2']:.6f} | "
            f"S(te) mae={te_s_reg_s['mae']:.6f} mse={te_s_reg_s['mse']:.6f} r2={te_s_reg_s['r2']:.6f} | "
            f"best_MAE={best_metric:.6f}{'*' if improved else ''}"
        )

    return {
        "y_key": y_key,
        "task": task,
        "out_dim": out_dim,
        "emb_dim": emb_dim,
        "epochs": epochs,
        "best_metric": best_metric,
        "student_ckpt": str(best_student_path),
        "teacher_ckpt": str(best_teacher_path),
        "norm_stats_3hop": norm_stats_3hop,
    }


if __name__ == "__main__":
    AUTO_ENCODE_ALL_CHUNKS = False  # train RAL with a train-only retrieval index
    train_designs = ["aes_cipher_top_1.0", "des_1.0", "spi_top_1.0"]
    test_designs = ["eth_top_1.0"]
    from utils.selected_cell import max_type_max_size
    maxType, max_size = max_type_max_size()

    hop_train_sigs = signatures_by_designs(train_designs, out_dir=chunk_dir)

    _norm_stats_cone, _norm_stats_3hop = build_norm_stats(
        train_designs,
        norm_fields_node=None,
        norm_fields_edge=None,
    )

    for y_key in ["slack_eco"]:
        train_one_task(
            y_key=y_key,
            device="cuda",
            hetero=False,
            gnn_type_student="gat",
            x_keys=["slack", "arrival", "trans", "ceff", "bbox", "level", "is_port", "is_outpin", "is_clock_network", "is_async_pin"],
            maxType=maxType,
            max_size=max_size,
            hid_dim=256,
            emb_dim=256,
            dropout=0.1,
            teacher_local_gnn="gcn",
            teacher_trans_layers=1,
            teacher_trans_heads=1,
            epochs=10,
            batch_size_cone=2,
            lr_teacher=2e-4,
            lr_student=2e-4,
            lambda_distill=0.5,
            train_designs=train_designs,
            test_designs=test_designs,
            norm_fields_node=None,
            norm_fields_edge=None,
            norm_stats_3hop=_norm_stats_3hop,
            norm_stats_cone=_norm_stats_cone,
            log_every=20,
        )

        print(f"[Done][{y_key}] student_ckpt={student_ckpt_path(y_key)}")
        print(f"[Done][{y_key}] teacher_ckpt={teacher_ckpt_path(y_key)}")

        build_retrieval_index(
            y_key=y_key,
            out_subdir="student3hop",
            index_split="train",
            device="cuda",
            hetero=False,
            gnn_type="gat",
            x_keys=["slack", "arrival", "trans", "ceff", "bbox", "level", "is_port", "is_outpin", "is_clock_network", "is_async_pin"],
            maxType=maxType,
            max_size=max_size,
            hid_dim=256,
            emb_dim=256,
            dropout=0.1,
            norm_stats_3hop=_norm_stats_3hop,
            signatures=None if AUTO_ENCODE_ALL_CHUNKS else hop_train_sigs,
            batch_size=128,
        )
