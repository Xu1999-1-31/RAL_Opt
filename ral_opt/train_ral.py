from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.env_setup import setup_env
from utils.ral_graph_data import ConeQueryDataset
from utils.chunk_graph_data import signatures_by_designs, build_train_norm_stats
from utils.eval_report import compute_regression_metrics, print_eval_report

from models.ral_opt import _infer_task
import dgl
from models.ral_decoder import TorchRetrievalIndex, RALCrossAttnDecoder

from data.Data_var import ep_dir as default_ep_dir
from data.Data_var import chunk_dir as default_chunk_dir
from ral_opt.ral_var import out_dir, teacher_ckpt_path
from ral_opt.pretrain_distill import load_teacher_from_ckpt


# ---------------------------------------------------------------------------
# Module-level dataclass and helpers
# ---------------------------------------------------------------------------

@dataclass
class CachedSample:
    """Pre-computed decoder-ready sample cached on CPU."""
    h_raw: torch.Tensor        # (N, cone_dim)
    teacher_yhat: torch.Tensor # (N, out_dim)
    outpin_nids: torch.Tensor  # (K,) int64
    z_retr: torch.Tensor       # (K, R, D)
    o_retr: torch.Tensor       # (K, R, O)
    retr_score: torch.Tensor   # (K, R)
    g_cone: Any                # CPU DGL homogeneous graph
    y: torch.Tensor            # (N, ...)
    design_id: str


def extract_cone_edges(g_cone: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (src, dst) int64 CPU tensors from a homo or hetero DGL cone graph."""
    if hasattr(g_cone, "ntypes") and len(g_cone.ntypes) > 1:
        all_src, all_dst = [], []
        for etype in g_cone.canonical_etypes:
            s, d = g_cone.edges(etype=etype)
            all_src.append(s.cpu())
            all_dst.append(d.cpu())
        if all_src:
            return torch.cat(all_src), torch.cat(all_dst)
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    src, dst = g_cone.edges()
    return src.cpu(), dst.cpu()


import tqdm

setup_env()


def retrieval_index_dir(
    *,
    y_key: str,
    chunk_dir: str = default_chunk_dir,
    out_subdir: str = "student3hop",
    task: Optional[str] = None,
    index_split: str = "train",
) -> str:
    """Return the retrieval index directory matching :func:`build_retrieval_index` layout."""
    t = str(task) if task is not None else str(_infer_task(y_key))
    out_subdir_task = f"{out_subdir}_{t}"
    return str(Path(chunk_dir) / "retrieval_index" / out_subdir_task / index_split / y_key)


def _save_ckpt(path: Path, *, model: torch.nn.Module, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"cfg": cfg, "state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def decoder_ckpt_path(y_key: str) -> Path:
    return Path(out_dir) / "ral_train" / y_key / "decoder.pt"


def _mask_for_labels(y: torch.Tensor) -> torch.Tensor:
    """Regression supervision mask: valid where finite and not sentinel -1."""
    if y.dim() == 1:
        return torch.isfinite(y) & (y != -1.0)
    return torch.isfinite(y).all(dim=1) & (y[:, 0] != -1.0)


class _RunningRegStats:
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


@torch.no_grad()
def _teacher_embeddings(
    teacher: torch.nn.Module,
    g_cone: Any,
    outpin_nids: torch.Tensor,
    dev: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (h_raw, node_z, outpin_z, node_yhat) from frozen teacher."""
    with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        node_yhat, node_z, outpin_z, h_raw = teacher(g_cone, outpin_nids=outpin_nids.to(dev))
    return h_raw.float(), node_z.float(), outpin_z.float(), node_yhat.float()


def train_ral_one_task(
    *,
    y_key: str,
    device: str,
    train_designs: List[str],
    test_designs: List[str],
    topk: int = 5,
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    ep_dir: str = default_ep_dir,
    chunk_dir: str = default_chunk_dir,
    out_subdir: str = "student3hop",
    index_split: str = "train",
    hetero: bool = True,
    x_keys: Optional[List[str]] = None,
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
    save_best: bool = True,
    log_every: int = 0,
    grad_clip: float = 1.0,
    lr_min: float = 1e-6,
    decoder_hid_dim: int = 256,
    decoder_dropout: float = 0.1,
    decoder_cross_attn_heads: int = 4,
    decoder_gnn_type: str = "gcn",
    decoder_gnn_layers: int = 2,
    decoder_gat_heads: int = 4,
    label_norm: bool = False,
    label_norm_ref_key: Optional[str] = "slack",
) -> Dict[str, Any]:
    """
    Single-task RAL training for regression targets (e.g. ``slack_eco``).

    Uses a cross-attention residual decoder on an augmented graph that joins
    each query cone with its retrieved outpins.  Retrieval results are
    pre-cached before the training loop so each epoch costs zero disk I/O.

    - Auto-infers out_dim and emb_dim from teacher checkpoint cfg.
    - Knowledge shielding: retrieval excludes same design_id for both train and test.
    """
    if x_keys is None:
        x_keys = ["bidirection_feature"]

    dev = torch.device(device)
    task = str(_infer_task(y_key))  # always "reg"

    train_sigs = signatures_by_designs(train_designs, out_dir=ep_dir)
    test_sigs = signatures_by_designs(test_designs, out_dir=ep_dir)

    ns_cone = build_train_norm_stats(
        train_signatures=train_sigs,
        out_dir=ep_dir,
        fields_node=norm_fields_node,
        fields_edge=norm_fields_edge,
        designs=train_designs,
    )

    from utils.chunk_graph_data import compute_per_design_label_stats as _compute_lns
    lns: Optional[Dict[str, Any]] = None
    if label_norm:
        all_designs = list(set(list(train_designs) + list(test_designs)))
        lns = _compute_lns(
            all_designs, out_dir=ep_dir, y_feat_key=y_key,
            label_norm_ref_key=label_norm_ref_key,
        )
        print(f"[RAL][label_norm] Per-design label stats (ref={label_norm_ref_key}):")
        for did, s in lns.items():
            print(f"  {did}: mean={s['mean']:.6f} std={s['std']:.6f} n={s['n']}")

    ds_train = ConeQueryDataset(
        ep_dir=ep_dir,
        cone_signatures=train_sigs,
        hetero=hetero,
        device=dev,
        normalize=True,
        norm_stats_cone=ns_cone,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        y_key=y_key,
    )
    ds_test = ConeQueryDataset(
        ep_dir=ep_dir,
        cone_signatures=test_sigs,
        hetero=hetero,
        device=dev,
        normalize=True,
        norm_stats_cone=ns_cone,
        norm_fields_node=norm_fields_node,
        norm_fields_edge=norm_fields_edge,
        y_key=y_key,
    )

    t_ckpt = str(teacher_ckpt_path(y_key))
    teacher = load_teacher_from_ckpt(y_key, device=device)
    emb_dim = teacher.emb_dim
    out_dim = teacher.out_dim
    for p in teacher.parameters():
        p.requires_grad_(False)

    idx_dir = retrieval_index_dir(
        y_key=y_key, chunk_dir=chunk_dir, out_subdir=out_subdir, task=task, index_split=index_split
    )
    index = TorchRetrievalIndex.load(index_dir=idx_dir, device=device)
    retr_val_dim = int(index.val.shape[1])

    cone_dim = int(teacher.hid_dim * 2)
    decoder = RALCrossAttnDecoder(
        cone_dim=cone_dim,
        emb_dim=emb_dim,
        out_dim=out_dim,
        retr_val_dim=retr_val_dim,
        task=task,
        hid_dim=decoder_hid_dim,
        dropout=decoder_dropout,
        cross_attn_heads=decoder_cross_attn_heads,
        gnn_type=decoder_gnn_type,
        gnn_num_layers=decoder_gnn_layers,
        gat_heads=decoder_gat_heads,
    ).to(dev)

    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)

    best_metric = float("inf")
    best_path = decoder_ckpt_path(y_key)
    global_step = 0

    def loss_fn(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m = _mask_for_labels(y.to(yhat.device))
        if m.sum().item() == 0:
            return yhat.sum() * 0.0
        return F.mse_loss(yhat[m], y.to(torch.float32).to(yhat.device)[m], reduction="mean")

    # ---------------------------------------------------------------
    # Pre-cache decoder-ready samples (CPU tensors / CPU DGL graphs).
    # ---------------------------------------------------------------
    _CachedSample = CachedSample

    def _build_cache(ds: Any, split: str) -> List[CachedSample]:
        cache: List[_CachedSample] = []
        for cidx in tqdm.tqdm(range(len(ds)), desc=f"[RAL] Pre-cache {split}", colour="cyan"):
            sample = ds[cidx]
            h_raw, node_z, outpin_z, t_yhat = _teacher_embeddings(
                teacher, sample.g_cone, sample.outpin_nids.to(dev), dev
            )
            op_nids = sample.outpin_nids.cpu()
            K = int(op_nids.shape[0])
            N = int(h_raw.shape[0])

            if K > 0:
                z_retr, o_retr, retr_score = index.search_batch(
                    queries=outpin_z, topk=topk, exclude_design_id=sample.design_id
                )
                z_retr = z_retr.cpu()
                o_retr = o_retr.cpu()
                retr_score = torch.where(
                    torch.isfinite(retr_score),
                    retr_score,
                    retr_score.new_full(retr_score.shape, -1e9),
                ).cpu()
            else:
                z_retr = torch.zeros((0, topk, emb_dim), dtype=torch.float32)
                o_retr = torch.zeros((0, topk, retr_val_dim), dtype=torch.float32)
                retr_score = torch.zeros((0, topk), dtype=torch.float32)

            cone_src, cone_dst = extract_cone_edges(sample.g_cone)
            g_cone_homo = dgl.graph((cone_src, cone_dst), num_nodes=N)

            y_cache = sample.y_cone_all.cpu()
            if lns is not None:
                s_ln = lns.get(sample.design_id, {"mean": 0.0, "std": 1.0})
                y_cache = (y_cache.float() - s_ln["mean"]) / s_ln["std"]

            cache.append(_CachedSample(
                h_raw=h_raw.cpu(),
                teacher_yhat=t_yhat.cpu(),
                outpin_nids=op_nids,
                z_retr=z_retr,
                o_retr=o_retr,
                retr_score=retr_score,
                g_cone=g_cone_homo,
                y=y_cache,
                design_id=sample.design_id,
            ))
        return cache

    train_cache = _build_cache(ds_train, "train")
    test_cache = _build_cache(ds_test, "test")
    index.to("cpu")
    teacher.cpu()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[RAL] Cache ready: train={len(train_cache)}, test={len(test_cache)} cones.")

    for ep in range(1, epochs + 1):
        # ---------------- train ----------------
        decoder.train()

        tr_loss = 0.0
        tr_steps = 0
        tr_valid_n = 0
        tr_reg = _RunningRegStats()

        train_indices = torch.randperm(len(train_cache)).tolist()
        for bidx, idx in enumerate(train_indices, start=1):
            global_step += 1
            s = train_cache[idx]
            opt.zero_grad(set_to_none=True)

            h_raw = s.h_raw.to(dev)
            t_yhat = s.teacher_yhat.to(dev)
            outpin_nids = s.outpin_nids.to(dev)
            z_retr = s.z_retr.to(dev)
            o_retr = s.o_retr.to(dev)
            retr_score = s.retr_score.to(dev)
            g_cone = s.g_cone.to(dev)
            y = s.y.to(dev)

            yhat = decoder(
                h_cone=h_raw,
                g_cone=g_cone,
                outpin_nids=outpin_nids,
                z_retr=z_retr,
                o_retr=o_retr,
                retr_score=retr_score,
                noise_std=noise_std,
                teacher_yhat=t_yhat,
            )
            total = loss_fn(yhat, y)
            if not torch.isfinite(total):
                print(f"[WARN][RAL] step {global_step}: loss is NaN/Inf, skipping")
                continue
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            opt.step()

            batch_loss = float(total.detach().cpu().item())
            if math.isfinite(batch_loss):
                tr_loss += batch_loss
                tr_steps += 1

            m = _mask_for_labels(y)
            batch_valid_n = int(m.sum().item()) if m.any() else 0
            tr_valid_n += batch_valid_n
            if m.any():
                yhat_d = yhat.detach()[m]
                y_d = y.detach()[m]
                if lns is not None:
                    s_ln = lns.get(s.design_id, {"mean": 0.0, "std": 1.0})
                    yhat_d = yhat_d * s_ln["std"] + s_ln["mean"]
                    y_d = y_d * s_ln["std"] + s_ln["mean"]
                tr_reg.update(yhat_d, y_d)

            if log_every > 0 and bidx % log_every == 0:
                _reg_b = tr_reg.finalize()
                print(
                    f"[Ep {ep:03d} | step {global_step:06d} | batch {bidx:04d}] "
                    f"loss={batch_loss:.6f} "
                    f"mae={_reg_b['mae']:.6f} mse={_reg_b['mse']:.6f} r2={_reg_b['r2']:.6f} "
                    f"valid_n={batch_valid_n}"
                )

        # ---------------- test ----------------
        decoder.eval()
        te_loss = 0.0
        te_steps = 0
        te_reg = _RunningRegStats()

        with torch.no_grad():
            for s in test_cache:
                h_raw = s.h_raw.to(dev)
                t_yhat = s.teacher_yhat.to(dev)
                outpin_nids = s.outpin_nids.to(dev)
                z_retr = s.z_retr.to(dev)
                o_retr = s.o_retr.to(dev)
                retr_score = s.retr_score.to(dev)
                g_cone = s.g_cone.to(dev)
                y = s.y.to(dev)

                yhat = decoder(
                    h_cone=h_raw,
                    g_cone=g_cone,
                    outpin_nids=outpin_nids,
                    z_retr=z_retr,
                    o_retr=o_retr,
                    retr_score=retr_score,
                    noise_std=0.0,
                    teacher_yhat=t_yhat,
                )
                _tl = float(loss_fn(yhat, y).detach().cpu().item())
                if math.isfinite(_tl):
                    te_loss += _tl
                    te_steps += 1

                m = _mask_for_labels(y)
                if m.any():
                    yhat_d = yhat.detach()[m]
                    y_d = y.detach()[m]
                    if lns is not None:
                        s_ln = lns.get(s.design_id, {"mean": 0.0, "std": 1.0})
                        yhat_d = yhat_d * s_ln["std"] + s_ln["mean"]
                        y_d = y_d * s_ln["std"] + s_ln["mean"]
                    te_reg.update(yhat_d, y_d)

        scheduler.step()

        tr_s = tr_reg.finalize()
        te_s = te_reg.finalize()
        metric = float(te_s["mae"])
        improved = metric < best_metric

        if improved and save_best:
            best_metric = metric
            _save_ckpt(
                best_path,
                model=decoder,
                cfg=dict(
                    decoder_type="crossattn_residual",
                    y_key=y_key,
                    task=task,
                    out_dim=out_dim,
                    cone_dim=cone_dim,
                    emb_dim=emb_dim,
                    retr_val_dim=retr_val_dim,
                    topk=topk,
                    teacher_ckpt=t_ckpt,
                    index_dir=idx_dir,
                    out_subdir=out_subdir,
                    index_split=index_split,
                    hid_dim=decoder_hid_dim,
                    cross_attn_heads=decoder_cross_attn_heads,
                    gnn_type=decoder_gnn_type,
                    gnn_num_layers=decoder_gnn_layers,
                ),
                extra=dict(best_metric=best_metric, epoch=ep),
            )

        print(
            f"[RAL][{y_key}][Ep {ep:03d}] "
            f"loss(tr)={tr_loss/max(tr_steps,1):.6f} loss(te)={te_loss/max(te_steps,1):.6f} | "
            f"mae(tr)={tr_s['mae']:.6f} mse(tr)={tr_s['mse']:.6f} r2(tr)={tr_s['r2']:.6f} | "
            f"mae(te)={te_s['mae']:.6f} mse(te)={te_s['mse']:.6f} r2(te)={te_s['r2']:.6f} | "
            f"valid_n(tr)={tr_valid_n} | "
            f"best_MAE={best_metric:.6f}{'*' if improved else ''}"
        )

    if save_best and best_path.exists():
        ckpt = torch.load(str(best_path), map_location=dev, weights_only=False)
        decoder.load_state_dict(ckpt["state_dict"])

    return {
        "y_key": y_key,
        "task": task,
        "out_dim": out_dim,
        "emb_dim": emb_dim,
        "retr_val_dim": retr_val_dim,
        "topk": topk,
        "epochs": epochs,
        "noise_std": noise_std,
        "best_metric": best_metric,
        "decoder_ckpt": str(best_path),
        "index_dir": idx_dir,
        "index_split": index_split,
        "teacher_ckpt": t_ckpt,
        "_decoder": decoder,
        "_test_cache": test_cache,
        "label_norm_stats": lns,
    }


@torch.no_grad()
def evaluate_per_design_ral(
    decoder: torch.nn.Module,
    test_cache: List,
    *,
    out_dim: int,
    y_key: str,
    noise_std: float = 0.0,
    label_norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Evaluate the decoder on each test design in the cache and print report."""
    decoder.eval()
    dev = next(decoder.parameters()).device

    groups: Dict[str, List] = {}
    for s in test_cache:
        groups.setdefault(s.design_id, []).append(s)

    per_design: Dict[str, Dict[str, float]] = {}
    all_preds, all_labels = [], []

    for design, samples in groups.items():
        d_preds, d_labels = [], []
        for s in samples:
            h_raw = s.h_raw.to(dev)
            g_cone = s.g_cone.to(dev)
            z_retr = s.z_retr.to(dev)
            o_retr = s.o_retr.to(dev)
            retr_score = s.retr_score.to(dev)
            op_nids = s.outpin_nids.to(dev)
            t_yhat = s.teacher_yhat.to(dev)

            yhat = decoder(
                h_cone=h_raw,
                g_cone=g_cone,
                outpin_nids=op_nids,
                z_retr=z_retr,
                o_retr=o_retr,
                retr_score=retr_score,
                noise_std=0.0,
                teacher_yhat=t_yhat,
            )
            yhat_cpu = yhat.cpu()
            y_cpu = s.y.clone()
            if label_norm_stats is not None and design in label_norm_stats:
                ls_d = label_norm_stats[design]
                yhat_cpu = yhat_cpu * ls_d["std"] + ls_d["mean"]
                y_cpu = y_cpu * ls_d["std"] + ls_d["mean"]
            d_preds.append(yhat_cpu)
            d_labels.append(y_cpu)

        if not d_preds:
            continue
        preds_cat = torch.cat(d_preds, dim=0)
        labels_cat = torch.cat(d_labels, dim=0)
        all_preds.append(preds_cat)
        all_labels.append(labels_cat)

        m = _mask_for_labels(labels_cat)
        per_design[design] = compute_regression_metrics(preds_cat[m], labels_cat[m])

    if not all_preds:
        print("[evaluate_per_design_ral] No data to evaluate.")
        return

    g_preds = torch.cat(all_preds, dim=0)
    g_labels = torch.cat(all_labels, dim=0)
    m = _mask_for_labels(g_labels)
    g_metrics = compute_regression_metrics(g_preds[m], g_labels[m])

    print_eval_report(y_key, per_design, g_metrics, header="RAL Best Decoder Evaluation")


if __name__ == "__main__":
    train_designs = ["aes_cipher_top_1.0", "des_1.0", "spi_top_1.0"]
    test_designs = ["eth_top_1.0"]

    y_keys = ["slack_eco"]

    for y_key in y_keys:
        train_ral_one_task(
            y_key=y_key,
            device="cuda",
            train_designs=train_designs,
            test_designs=test_designs,
            topk=5,
            epochs=10,
            batch_size=1,
            lr=2e-4,
            weight_decay=0.0,
            noise_std=0.01,
            ep_dir=default_ep_dir,
            chunk_dir=default_chunk_dir,
            out_subdir="student3hop",
            index_split="train",
            hetero=False,
            x_keys=["slack", "arrival", "trans", "ceff", "bbox", "level",
                    "is_port", "is_outpin", "is_clock_network", "is_async_pin"],
            log_every=20,
            decoder_hid_dim=256,
            decoder_dropout=0.1,
            decoder_cross_attn_heads=4,
            decoder_gnn_type="gcn",
            decoder_gnn_layers=2,
        )

        print(f"[Done][{y_key}] decoder_ckpt={decoder_ckpt_path(y_key)}")
