import os
from pathlib import Path

ral_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(ral_dir, "checkpoints")


def student_ckpt_path(y_key: str, *, checkpoint_dir: str = out_dir) -> Path:
    """Return the default best-student checkpoint path for *y_key*.

    Layout: ``<checkpoint_dir>/ral_distill/<y_key>/student_best.pt``
    """
    return Path(checkpoint_dir) / "ral_distill" / y_key / "student_best.pt"


def teacher_ckpt_path(y_key: str, *, checkpoint_dir: str = out_dir) -> Path:
    """Return the default best-teacher checkpoint path for *y_key*.

    Layout: ``<checkpoint_dir>/ral_distill/<y_key>/teacher_best.pt``
    """
    return Path(checkpoint_dir) / "ral_distill" / y_key / "teacher_best.pt"