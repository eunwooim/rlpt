import logging
import os
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"is_fx_tracing will return true.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"Kwargs passed to `processor.__call__`.*",
    category=UserWarning,
)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._export").setLevel(logging.ERROR)

# More stable for DataLoader workers that pass image tensors / PIL-derived tensors.
try:
    import torch.multiprocessing as mp

    strategy = os.environ.get("TORCH_MP_SHARING_STRATEGY", "file_system")
    mp.set_sharing_strategy(strategy)
except Exception:
    pass
