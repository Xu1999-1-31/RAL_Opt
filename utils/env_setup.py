import warnings
import random
import os
import numpy as np
import torch


def setup_env(seed: int = 42, deterministic: bool = True):
    """Setup environment for reproducibility and warning control."""
    warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.*` is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning, module="dgl")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    os.environ["PYTHONHASHSEED"] = str(seed)
