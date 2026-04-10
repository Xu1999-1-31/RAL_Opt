"""Shared per-design and global regression evaluation utilities.

Provides metric computation and pretty-printing for regression tasks.
Used by RAL training.
"""
from __future__ import annotations

from typing import Dict

import torch


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """MAE, MSE, R² from flat (N,) tensors."""
    p = y_pred.detach().to("cpu", dtype=torch.float64).view(-1)
    t = y_true.detach().to("cpu", dtype=torch.float64).view(-1)
    fin = torch.isfinite(p) & torch.isfinite(t)
    p = p[fin]
    t = t[fin]
    diff = p - t
    n = int(t.numel())
    if n == 0:
        return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "n": 0}
    mae = float(diff.abs().mean().item())
    mse = float((diff * diff).mean().item())
    sse = float((diff * diff).sum().item())
    sst = float(((t - t.mean()) ** 2).sum().item())
    r2 = float("nan") if sst <= 1e-12 else float(1.0 - sse / sst)
    return {"mae": mae, "mse": mse, "r2": r2, "n": n}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

_SEP = "=" * 64
_SUB = "-" * 48


def print_eval_report(
    y_key: str,
    per_design: Dict[str, Dict[str, float]],
    global_metrics: Dict[str, float],
    header: str = "Evaluation Report",
) -> None:
    print(f"\n{_SEP}")
    print(f"  {header}")
    print(f"  y_key={y_key}")
    print(_SEP)

    for design, m in per_design.items():
        print(f"\n  {_SUB}")
        print(f"  Design: {design}   (N={m.get('n', '?')})")
        print(f"  {_SUB}")
        _print_reg(m, indent=4)

    print(f"\n  {_SUB}")
    print(f"  GLOBAL (all test designs)   (N={global_metrics.get('n', '?')})")
    print(f"  {_SUB}")
    _print_reg(global_metrics, indent=4)

    print(f"\n{_SEP}\n")


def _print_reg(m: Dict[str, float], indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}MAE   = {m.get('mae', float('nan')):.6f}")
    print(f"{pad}MSE   = {m.get('mse', float('nan')):.6f}")
    print(f"{pad}R²    = {m.get('r2', float('nan')):.6f}")
