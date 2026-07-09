import torch
import logging
import random
import warnings
import numpy as np
from dataclasses import dataclass


def setup_env():
    """Sets up the environment for training/inference."""
    # Suppress known external warnings
    warnings.filterwarnings(
        "ignore", message=".*flex_attention called without torch.compile.*"
    )
    warnings.filterwarnings("ignore", message=".*HF_HUB_ENABLE_HF_TRANSFER.*")
    warnings.filterwarnings(
        "ignore", message=".*TensorFloat32 tensor cores.*not enabled.*"
    )

    # Prevent TF32 precision loss in router sigmoid gate
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = True

    import torch._dynamo.config as dynamo_config
    import torch.compiler.config as compiler_config

    if hasattr(dynamo_config, "inline_inbuilt_nn_modules"):
        dynamo_config.inline_inbuilt_nn_modules = True

    if hasattr(dynamo_config, "enable_compiler_collectives"):
        dynamo_config.enable_compiler_collectives = True

    if hasattr(compiler_config, "allow_unspec_int_on_nn_module"):
        compiler_config.allow_unspec_int_on_nn_module = True
    if hasattr(compiler_config, "assume_static_by_default"):
        compiler_config.assume_static_by_default = False
    if hasattr(compiler_config, "automatic_dynamic_shapes"):
        compiler_config.automatic_dynamic_shapes = True
    if hasattr(dynamo_config, "assume_dunder_attributes_remain_unchanged"):
        dynamo_config.assume_dunder_attributes_remain_unchanged = True
    # Capture remaining scalar tensor conversions
    if hasattr(dynamo_config, "capture_scalar_outputs"):
        dynamo_config.capture_scalar_outputs = True


def get_device() -> torch.device:
    """Returns the available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ColorFormatter(logging.Formatter):
    """ANSI color formatter for logger output."""

    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Color mappings
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.INFO: blue
        + "%(asctime)s | "
        + green
        + "%(levelname)s"
        + reset
        + " | %(message)s",
        logging.WARNING: yellow + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s | %(levelname)s | %(message)s" + reset,
        logging.CRITICAL: bold_red
        + "%(asctime)s | %(levelname)s | %(message)s"
        + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(name: str = __name__) -> logging.Logger:
    """Configures and returns a colored logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(ColorFormatter())
        logger.addHandler(ch)

    # Silence excessive deep-compilation Inductor warnings if present
    logging.getLogger("torch._inductor").setLevel(logging.ERROR)

    return logger


def seed_everything(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class HSPMNConfig:
    """Configuration for HSPMN v3.0 model."""

    dim: int = 768
    num_heads: int = 12
    num_kv_heads: int = 4
    sparsity_k: float = 0.2
    mlp_ratio: int = 4
    max_seq_len: int = 16384
    rope_base: int = 10000
    router_sparsity_coef: float = 0.1
    router_entropy_coef: float = 0.01
    num_sink_tokens: int = 128

    def __post_init__(self):
        assert self.dim % self.num_heads == 0, "Dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, (
            "Heads must be divisible by KV heads (GQA)"
        )
        self.head_dim = self.dim // self.num_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
