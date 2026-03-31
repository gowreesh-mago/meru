#!/usr/bin/env python3
"""
Comprehensive import check to verify all dependencies are installed
Run this before starting experiments to catch missing packages early

This checks ALL packages used across the MERU project:
- Core ML libraries (torch, torchvision, numpy)
- Vision models (timm)
- Config/logging (omegaconf, hydra, loguru)
- Metrics/evaluation (sklearn, torchmetrics)
- Text processing (ftfy, regex)
- Data loading (webdataset, wordsegment - optional)
- Project modules (meru, emoset)
"""

import sys
from pathlib import Path

# Color codes for terminal output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_import(module_name, package_name=None, optional=False):
    """Try to import a module and report status"""
    try:
        __import__(module_name)
        status = f"{GREEN}✓{RESET}"
        if optional:
            status = f"{BLUE}✓{RESET}"
        print(f"{status} {module_name}")
        return True
    except ImportError as e:
        pkg = package_name or module_name
        if optional:
            print(f"{YELLOW}○{RESET} {module_name} (optional) - Install with: pip install {pkg}")
            return True  # Don't fail for optional packages
        else:
            print(f"{RED}✗{RESET} {module_name} - Install with: pip install {pkg}")
            print(f"  Error: {str(e)[:80]}")
            return False

def check_cuda():
    """Check CUDA availability and version"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{GREEN}✓{RESET} CUDA available")
            print(f"  Version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{YELLOW}○{RESET} CUDA not available (CPU-only mode)")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} Error checking CUDA")
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 70)
    print("COMPREHENSIVE DEPENDENCY CHECK - MERU PROJECT")
    print("=" * 70)
    print()
    print("Legend:")
    print(f"  {GREEN}✓{RESET} Required package found")
    print(f"  {BLUE}✓{RESET} Optional package found")
    print(f"  {YELLOW}○{RESET} Optional package missing (not critical)")
    print(f"  {RED}✗{RESET} Required package missing (must install!)")
    print()

    all_ok = True

    # Core ML dependencies (REQUIRED)
    print("Core ML dependencies (REQUIRED):")
    all_ok &= check_import("torch")
    all_ok &= check_import("torchvision")
    all_ok &= check_import("numpy")
    all_ok &= check_import("PIL", "Pillow")
    print()

    # Vision model dependencies (REQUIRED)
    print("Vision model dependencies (REQUIRED):")
    all_ok &= check_import("timm")
    print()

    # Config and logging (REQUIRED)
    print("Config and logging (REQUIRED):")
    all_ok &= check_import("omegaconf")
    all_ok &= check_import("hydra", "hydra-core")
    all_ok &= check_import("loguru")
    all_ok &= check_import("yaml", "pyyaml")
    print()

    # Training dependencies (REQUIRED)
    print("Training dependencies (REQUIRED):")
    all_ok &= check_import("tqdm")
    all_ok &= check_import("tensorboard")
    print()

    # Evaluation and metrics (REQUIRED)
    print("Evaluation and metrics (REQUIRED):")
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("torchmetrics")
    print()

    # Text processing (REQUIRED for tokenizer)
    print("Text processing (REQUIRED):")
    all_ok &= check_import("ftfy")
    all_ok &= check_import("regex")
    print()

    # Data loading (OPTIONAL - only needed for RedCaps training)
    print("Data loading (OPTIONAL - only for RedCaps training):")
    check_import("webdataset", optional=True)
    check_import("wordsegment", optional=True)
    print()

    # Visualization (OPTIONAL - useful for analysis)
    print("Visualization (OPTIONAL):")
    check_import("matplotlib", optional=True)
    print()

    # CUDA check
    print("GPU/CUDA:")
    check_cuda()
    print()

    # Check project modules (REQUIRED)
    print("Project modules (REQUIRED):")
    try:
        # Add repo root to path
        repo_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(repo_root))

        all_ok &= check_import("meru")
        all_ok &= check_import("meru.data")
        all_ok &= check_import("meru.models")
        all_ok &= check_import("meru.tokenizer")

        # Check emoset module
        sys.path.insert(0, str(repo_root / "emoset"))
        all_ok &= check_import("Emoset", "emoset")
    except Exception as e:
        print(f"{RED}✗{RESET} Error importing project modules")
        print(f"  Error: {e}")
        all_ok = False
    print()

    # Check emotion integration
    print("Emotion classification modules:")
    try:
        from meru.emotion import CLIPCoOpEmotion, MERUCoOpEmotion
        print(f"{GREEN}✓{RESET} meru.emotion models")
    except Exception as e:
        print(f"{RED}✗{RESET} meru.emotion models failed")
        print(f"  Error: {str(e)[:80]}")
        all_ok = False
    print()

    print("=" * 70)
    if all_ok:
        print(f"{GREEN}✓✓✓ ALL REQUIRED DEPENDENCIES SATISFIED ✓✓✓{RESET}")
        print("=" * 70)
        print()
        print("You're ready to run experiments!")
        print()
        return 0
    else:
        print(f"{RED}✗✗✗ MISSING REQUIRED DEPENDENCIES ✗✗✗{RESET}")
        print("=" * 70)
        print()
        print("Install ALL required packages with:")
        print()
        print("  # Core ML and vision")
        print("  pip install torch torchvision numpy Pillow timm")
        print()
        print("  # Config, logging, training")
        print("  pip install omegaconf hydra-core loguru pyyaml tqdm tensorboard")
        print()
        print("  # Evaluation and text processing")
        print("  pip install scikit-learn torchmetrics ftfy regex")
        print()
        print("Optional packages (for RedCaps training or visualization):")
        print("  pip install webdataset wordsegment matplotlib")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
