#!/usr/bin/env python3
"""RAL-Opt unified orchestrator.

Usage:
    python main.py config/chunk.json     # chunk & store timing graphs
    python main.py config/distill.json   # teacher-student distillation + retrieval index
    python main.py config/ral.json       # RAL decoder training

Global settings (y_keys, designs, x_keys, device) are read from
config/global.json automatically.  Stage configs can override them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.env_setup import setup_env
setup_env()

logger = logging.getLogger("RAL_Opt.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_global_config(global_path: Optional[str] = None) -> Dict[str, Any]:
    if global_path is None:
        global_path = str(_PROJECT_ROOT / "config" / "global.json")
    return _load_json(global_path)


def _merge_global(stage_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_cfg)
    merged.update({k: v for k, v in stage_cfg.items() if not k.startswith("_")})
    return merged


def _ensure_list(val: Any) -> Optional[List]:
    if val is None:
        return None
    if isinstance(val, list):
        return val
    return [val]


# ── Stage: chunk & store ─────────────────────────────────────────────

def _run_chunk(cfg: Dict[str, Any]) -> None:
    from work.work_var import design_dir
    from data.Chunk_Store import (
        build_timing_graph_and_export,
        build_timing_graph_and_export_ep_cones,
        build_packed_timing_graph,
    )

    designs = _ensure_list(cfg.get("designs")) or list(design_dir.keys())
    c3 = cfg.get("chunk_3hop", {})
    cep = cfg.get("chunk_ep_cones", {})

    for design in designs:
        logger.info(f"[chunk] Building TimingGraph for {design}")
        t_tg = time.time()
        try:
            TG, packed = build_packed_timing_graph(design)
        except Exception as e:
            logger.error(f"[chunk] TimingGraph build FAIL for {design}: {e}")
            continue
        logger.info(f"[chunk] TimingGraph ready ({time.time()-t_tg:.1f}s)")

        t0 = time.time()
        try:
            build_timing_graph_and_export(
                design, neg_only=c3.get("neg_only", True),
                export_kwargs={
                    "k_hop": c3.get("k_hop", 3), "num_workers": c3.get("num_workers", 16),
                    "chunksize": c3.get("chunksize", 64), "write_node_names": c3.get("write_node_names", True),
                },
                TG=TG, packed=packed,
            )
            logger.info(f"[chunk] 3-hop  {design} done ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"[chunk] 3-hop  {design} FAIL: {e}")

        t0 = time.time()
        try:
            build_timing_graph_and_export_ep_cones(
                design, TG=TG, packed=packed,
                export_kwargs={
                    "cone_dir": cep.get("cone_dir", "fanin"), "max_hops": cep.get("max_hops"),
                    "max_nodes": cep.get("max_nodes", 100000), "num_workers": cep.get("num_workers", 8),
                    "write_node_names": cep.get("write_node_names", True),
                },
            )
            logger.info(f"[chunk] EP-cone {design} done ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.error(f"[chunk] EP-cone {design} FAIL: {e}")

        del TG, packed


# ── Stage: distill + retrieval index ─────────────────────────────────

def _run_distill(cfg: Dict[str, Any]) -> None:
    from ral_opt.pretrain_distill import train_one_task, build_retrieval_index, build_norm_stats, signatures_by_designs
    from ral_opt.ral_var import student_ckpt_path, teacher_ckpt_path
    from data.Data_var import chunk_dir
    from utils.selected_cell import max_type_max_size

    global_cfg = _load_global_config()
    y_keys = cfg.get("y_keys") or global_cfg.get("y_keys", ["slack_eco"])
    train_designs = cfg.get("train_designs") or global_cfg["train_designs"]
    test_designs = cfg.get("test_designs") or global_cfg["test_designs"]
    x_keys = cfg.get("x_keys") or global_cfg["x_keys"]
    device = cfg.get("device") or global_cfg.get("device", "cuda")

    maxType, max_size = max_type_max_size()
    hop_train_sigs = signatures_by_designs(train_designs, out_dir=chunk_dir)
    _norm_cone, _norm_3hop = build_norm_stats(train_designs)

    for y_key in y_keys:
        logger.info(f"[distill] START y_key={y_key}")
        t0 = time.time()
        train_one_task(
            y_key=y_key, device=device, hetero=cfg.get("hetero", False),
            gnn_type_student=cfg.get("gnn_type_student", "gat"), x_keys=x_keys,
            maxType=maxType, max_size=max_size,
            hid_dim=cfg.get("hid_dim", 256), emb_dim=cfg.get("emb_dim", 256),
            dropout=cfg.get("dropout", 0.1),
            teacher_local_gnn=cfg.get("teacher_local_gnn", "gcn"),
            teacher_trans_layers=cfg.get("teacher_trans_layers", 1),
            teacher_trans_heads=cfg.get("teacher_trans_heads", 1),
            epochs=cfg.get("epochs", 10), batch_size_cone=cfg.get("batch_size_cone", 2),
            lr_teacher=cfg.get("lr_teacher", 2e-4), lr_student=cfg.get("lr_student", 2e-4),
            lambda_distill=cfg.get("lambda_distill", 0.5),
            train_designs=train_designs, test_designs=test_designs,
            norm_stats_cone=_norm_cone, norm_stats_3hop=_norm_3hop,
            log_every=cfg.get("log_every", 20),
            label_norm=cfg.get("label_norm", y_key == "slack_eco"),
            label_norm_ref_key=cfg.get("label_norm_ref_key", "slack"),
        )
        logger.info(f"[distill] DONE  y_key={y_key} ({time.time()-t0:.1f}s)")
        logger.info(f"  student_ckpt = {student_ckpt_path(y_key)}")
        logger.info(f"  teacher_ckpt = {teacher_ckpt_path(y_key)}")

        if cfg.get("build_index_after_train", True):
            logger.info(f"[distill] Building retrieval index for y_key={y_key}")
            build_retrieval_index(
                y_key=y_key, out_subdir=cfg.get("index_out_subdir", "student3hop"),
                index_split=cfg.get("index_split", "train"), device=device,
                hetero=cfg.get("hetero", False), gnn_type=cfg.get("gnn_type_student", "gat"),
                x_keys=x_keys, maxType=maxType, max_size=max_size,
                hid_dim=cfg.get("hid_dim", 256), emb_dim=cfg.get("emb_dim", 256),
                dropout=cfg.get("dropout", 0.1), norm_stats_3hop=_norm_3hop,
                signatures=hop_train_sigs, batch_size=cfg.get("index_batch_size", 128),
            )
            logger.info(f"[distill] Retrieval index built for y_key={y_key}")


# ── Stage: RAL training ──────────────────────────────────────────────

def _run_ral(cfg: Dict[str, Any]) -> None:
    from ral_opt.train_ral import train_ral_one_task, decoder_ckpt_path, evaluate_per_design_ral

    global_cfg = _load_global_config()
    y_keys = cfg.get("y_keys") or global_cfg.get("y_keys", ["slack_eco"])
    train_designs = cfg.get("train_designs") or global_cfg["train_designs"]
    test_designs = cfg.get("test_designs") or global_cfg["test_designs"]
    x_keys = cfg.get("x_keys") or global_cfg["x_keys"]
    device = cfg.get("device") or global_cfg.get("device", "cuda")

    for y_key in y_keys:
        logger.info(f"[ral] START y_key={y_key}")
        t0 = time.time()
        result = train_ral_one_task(
            y_key=y_key, device=device, train_designs=train_designs, test_designs=test_designs,
            topk=cfg.get("topk", 5), epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 1), lr=cfg.get("lr", 2e-4),
            weight_decay=cfg.get("weight_decay", 0.0), noise_std=cfg.get("noise_std", 0.01),
            out_subdir=cfg.get("index_out_subdir", "student3hop"),
            index_split=cfg.get("index_split", "train"), hetero=cfg.get("hetero", False),
            x_keys=x_keys, log_every=cfg.get("log_every", 20),
            grad_clip=cfg.get("grad_clip", 1.0), lr_min=cfg.get("lr_min", 1e-6),
            decoder_hid_dim=cfg.get("decoder_hid_dim", 256),
            decoder_dropout=cfg.get("decoder_dropout", 0.1),
            decoder_cross_attn_heads=cfg.get("decoder_cross_attn_heads", 4),
            decoder_gnn_type=cfg.get("decoder_gnn_type", "gcn"),
            decoder_gnn_layers=cfg.get("decoder_gnn_layers", 2),
            decoder_gat_heads=cfg.get("decoder_gat_heads", 4),
            label_norm=cfg.get("label_norm", y_key == "slack_eco"),
            label_norm_ref_key=cfg.get("label_norm_ref_key", "slack"),
        )
        logger.info(f"[ral] DONE  y_key={y_key} ({time.time()-t0:.1f}s)")
        logger.info(f"  decoder_ckpt = {decoder_ckpt_path(y_key)}")

        evaluate_per_design_ral(
            result["_decoder"], result["_test_cache"],
            out_dim=result["out_dim"], y_key=y_key,
            label_norm_stats=result.get("label_norm_stats"),
        )


# ── Dispatcher ────────────────────────────────────────────────────────

_STAGE_MAP = {"chunk": _run_chunk, "distill": _run_distill, "ral": _run_ral}
_FILE_TO_STAGE = {"chunk.json": "chunk", "distill.json": "distill", "ral.json": "ral"}


def _detect_stage(config_path: str) -> str:
    basename = os.path.basename(config_path)
    stage = _FILE_TO_STAGE.get(basename)
    if stage is not None:
        return stage
    raise ValueError(f"Cannot detect stage from '{basename}'. Expected: {list(_FILE_TO_STAGE.keys())}")


def main():
    parser = argparse.ArgumentParser(description="RAL-Opt unified orchestrator", epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to stage config JSON (e.g. config/distill.json)")
    parser.add_argument("--global-config", default=None, help="Path to global config JSON (default: config/global.json)")
    parser.add_argument("--stage", choices=list(_STAGE_MAP.keys()), default=None, help="Force stage type")
    args = parser.parse_args()

    stage_cfg = _load_json(args.config)
    global_cfg = _load_global_config(args.global_config)
    merged = _merge_global(stage_cfg, global_cfg)

    stage = args.stage or _detect_stage(args.config)
    runner = _STAGE_MAP[stage]

    logger.info(f"{'='*60}")
    logger.info(f"RAL-Opt  stage={stage}  config={args.config}")
    logger.info(f"{'='*60}")

    t0 = time.time()
    runner(merged)
    logger.info(f"Stage [{stage}] finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
