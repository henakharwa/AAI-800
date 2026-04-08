"""
LCLDD Environment Setup & Verification Script
Run this once after installing requirements to verify everything is configured correctly.

Usage:
    py setup_env.py
"""

import sys
import importlib
import os

# ── Required packages and minimum versions ────────────────────────────────────
REQUIRED = {
    "torch":         "2.3.0",
    "transformers":  "4.43.0",
    "datasets":      "2.20.0",
    "accelerate":    "0.31.0",
    "wandb":         "0.17.0",
    "numpy":         "1.26.0",
    "scipy":         "1.13.0",
    "tqdm":          "4.66.0",
    "yaml":          None,    # pyyaml, no version check needed
    "dotenv":        None,    # python-dotenv
}

OPTIONAL = {
    "deepspeed": "0.14.4",   # Linux/WSL only
}


def check_version(pkg_name, actual_version, min_version):
    if min_version is None:
        return True
    try:
        from packaging.version import Version
        return Version(actual_version) >= Version(min_version)
    except Exception:
        # packaging not installed or version unparseable — assume ok
        return True


def verify_packages():
    print("=" * 60)
    print("  LCLDD Environment Verification")
    print("=" * 60)
    print(f"  Python: {sys.version}")
    print("=" * 60)

    all_ok = True

    print("\n[Required Packages]")
    for pkg, min_ver in REQUIRED.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            ok = check_version(pkg, version, min_ver)
            status = "OK" if ok else "OUTDATED"
            min_str = f"(need >={min_ver})" if min_ver else ""
            print(f"  {'[OK]' if ok else '[!!]'}  {pkg:<20} {version:<12} {min_str}")
            if not ok:
                all_ok = False
        except ImportError:
            print(f"  [MISSING]  {pkg}")
            all_ok = False

    print("\n[Optional Packages]")
    for pkg, min_ver in OPTIONAL.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]  {pkg:<20} {version}")
        except ImportError:
            print(f"  [--]  {pkg:<20} not installed (Linux/WSL only)")

    print("\n[PyTorch Details]")
    try:
        import torch
        print(f"  CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version   : {torch.version.cuda}")
            print(f"  GPU count      : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}          : {name} ({mem:.1f} GB)")
        else:
            print("  Running on CPU only (GPU training will require CUDA)")
    except ImportError:
        print("  torch not installed")

    print("\n[Project Directories]")
    dirs = ["configs", "data", "outputs", "cache", "src", "logs"]
    for d in dirs:
        exists = os.path.isdir(d)
        print(f"  {'[OK]' if exists else '[--]'}  {d}/")

    print("\n[W&B Config]")
    wandb_key = os.environ.get("WANDB_API_KEY", None)
    if wandb_key:
        print(f"  WANDB_API_KEY  : set (****{wandb_key[-4:]})")
    else:
        print("  WANDB_API_KEY  : NOT SET — add to .env file")

    print("\n" + "=" * 60)
    if all_ok:
        print("  All required packages installed. Environment ready.")
    else:
        print("  Some packages missing. Run: py -m pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    verify_packages()
