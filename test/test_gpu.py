"""
Test script to check GPU availability and configuration.
"""

import platform
import torch
import sys


def check_gpu_config():
    """Check GPU configuration and availability."""
    print("\nSystem Information:")
    print("-" * 50)
    print(f"Platform: {platform.system()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")

    print("\nPyTorch Configuration:")
    print("-" * 50)
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        try:
            # Test MPS with a small tensor operation
            device = torch.device("mps")
            x = torch.ones(1, device=device)
            print("\nMPS Test:")
            print("-" * 50)
            print(f"Test tensor created on MPS: {x}")
            print("MPS is working correctly!")
        except Exception as e:
            print(f"\nError testing MPS: {str(e)}")
    else:
        print("\nMPS is not available on this system")


if __name__ == "__main__":
    check_gpu_config()
