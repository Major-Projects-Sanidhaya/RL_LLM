"""
Training Utilities Module

This module provides helper functions for training setup and reproducibility.

Functions:
- set_seed(): Sets random seeds for reproducible experiments
- get_device(): Detects and returns appropriate device (CUDA/CPU)

How it works:
- set_seed() configures all random number generators (Python, NumPy, PyTorch)
- get_device() checks for GPU availability and returns appropriate torch.device

Implementation approach:
- Seeds are set globally for reproducibility across runs
- CUDA seeds are set for both single and multi-GPU setups
- Device detection enables seamless CPU/GPU switching
"""

import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all devices)


def get_device():
    """
    Get the appropriate device for training (CUDA if available, else CPU)

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')