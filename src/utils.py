"""Shared constants and helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NCES_DIR = RAW_DIR / "nces"
IPEDS_DIR = RAW_DIR / "ipeds"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
CORPUS_DIR = DATA_DIR / "corpus"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
OUTPUT_DIR = DATA_DIR / "output"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed every RNG that touches the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
