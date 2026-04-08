"""
Step 3 — Dataset Preparation & Teacher Trajectory Extraction

Downloads all datasets, runs teacher inference, extracts hidden-state
trajectories, motion vectors, and Jacobians. Saves to disk.

Usage:
    # Quick smoke test (5 samples from GSM8K only):
    py -3.12 prepare_data.py --sample 5 --datasets gsm8k

    # Full extraction for one dataset:
    py -3.12 prepare_data.py --datasets gsm8k

    # Full extraction for all datasets:
    py -3.12 prepare_data.py
"""

import os
import sys
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(__file__))

from src.data.datasets import load_all_datasets
from src.data.teacher_trajectories import TrajectoryExtractor
from src.data.cache_manager import TrajectoryCache
from src.models.loader import load_teacher


def parse_args():
    parser = argparse.ArgumentParser(description="LCLDD data preparation")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "math", "bbh", "arc_challenge"],
        help="Which datasets to process",
    )
    parser.add_argument(
        "--teacher", default="qwen2.5-7b",
        choices=["qwen2.5-7b", "llama3.1-8b"],
        help="Which teacher model to use",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Process only N samples per dataset (for testing)",
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "test"],
        help="Which split to process",
    )
    parser.add_argument(
        "--cache-dir", default="./cache/huggingface",
        help="HuggingFace model/dataset cache",
    )
    parser.add_argument(
        "--traj-dir", default="./data/trajectories",
        help="Where to save trajectory files",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip datasets that already have cached trajectories",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Samples per GPU generation call (default 4, increase if VRAM allows)",
    )
    parser.add_argument(
        "--max-per-dataset", type=int, default=None,
        help="Cap samples per dataset (e.g. 2000) to finish faster",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Main] Device: {device}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")

    # ── Step 3a: Load all datasets ─────────────────────────────────────────
    print("\n[Step 3a] Loading datasets...")
    all_data = load_all_datasets(
        cache_dir=args.cache_dir,
        datasets=args.datasets,
    )

    # ── Step 3b: Load teacher model ────────────────────────────────────────
    print(f"\n[Step 3b] Loading teacher: {args.teacher}")
    teacher_bundle = load_teacher(
        args.teacher,
        cache_dir=args.cache_dir,
        quantize=True,
        gpu_memory_gb=8,
    )

    # ── Step 3c: Set up extractor and cache ────────────────────────────────
    extractor = TrajectoryExtractor(
        model=teacher_bundle.model,
        tokenizer=teacher_bundle.tokenizer,
        max_new_tokens=256,
        max_steps=8,
        batch_size=args.batch_size,
        device=device,
    )

    cache = TrajectoryCache(args.traj_dir)
    cache.summary()

    # ── Step 3d: Extract trajectories per dataset ──────────────────────────
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  Processing: {dataset_name} ({args.split} split)")
        print(f"{'='*60}")

        if args.skip_existing and cache.exists(dataset_name, args.split):
            print(f"  [Skip] Already cached. Use --skip-existing=False to rerun.")
            continue

        samples = all_data[dataset_name][args.split]

        if args.sample is not None:
            samples = samples[:args.sample]
            print(f"  [Sample mode] Using first {len(samples)} samples")
        elif args.max_per_dataset is not None:
            samples = samples[:args.max_per_dataset]
            print(f"  [Capped] Using first {len(samples)} samples")

        print(f"  Total samples to process: {len(samples)}")

        # Extract trajectories (filter + extract hidden states + motion + Jacobian)
        checkpoint_path = f"{args.traj_dir}/{dataset_name}_{args.split}_checkpoint.pt"
        trajectories = extractor.extract_batch(
            samples,
            show_progress=True,
            save_every=200,
            checkpoint_path=checkpoint_path,
        )

        if trajectories:
            cache.save(dataset_name, args.split, trajectories)
        else:
            print(f"  [Warning] No trajectories kept for {dataset_name}!")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    cache.summary()
    print("\n[Done] Data preparation complete.")


if __name__ == "__main__":
    main()
