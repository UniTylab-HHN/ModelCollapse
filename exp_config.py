import os
import random
from typing import Optional

import numpy as np
import torch


# Base model name used in Experiment 4
BASE_MODEL: str = "aaditya/Llama3-OpenBioLLM-8B"

# Experiment parameters
NUM_GENERATIONS: int = 10
NUM_HUMAN_DOCS: int = 800
NUM_SYNTHETIC_PER_GEN: int = 800

# Token settings
REAL_TOKENS: int = 40
SYNTHETIC_TOKENS: int = 120
TOTAL_TOKENS: int = REAL_TOKENS + SYNTHETIC_TOKENS

# Training settings
LEARNING_RATE: float = 2e-4
NUM_TRAIN_EPOCHS: int = 5
BATCH_SIZE: int = 2
GRADIENT_ACCUMULATION: int = 8

# Random seed for reproducibility
SEED: int = 42

# Output directories
EXPERIMENT_NAME: str = "autophagy_no_filter"
OUTPUT_DIR: str = f"experiment_{EXPERIMENT_NAME}"
MODELS_DIR: str = os.path.join(OUTPUT_DIR, "models")
DATA_DIR: str = os.path.join(OUTPUT_DIR, "data")


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    This helps make the experiment results reproducible.
    """
    if seed is None:
        seed = SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs() -> None:
    """
    Create the main output, models, and data directories if they do not exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

