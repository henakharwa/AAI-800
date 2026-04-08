"""
LCLDD Main Training Script — runs curriculum Stages A through G.

Usage:
    # Smoke test — 20 steps per stage, fast verify
    py -3.12 train.py --stages A B --max-steps 20 --eval-every 10

    # Single stage
    py -3.12 train.py --stages A --max-steps 500 --eval-every 100

    # Full curriculum (all 7 stages)
    py -3.12 train.py --stages A B C D E F G

    # Resume from stage D
    py -3.12 train.py --stages D E F G --resume outputs/checkpoints/stage_C_step1000.pt
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

from src.models.loader import load_student
from src.student import LCLDDStudent
from src.training import Trainer, build_dataloaders, STAGE_CONFIGS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stages",       nargs="+", default=["A","B","C","D","E","F","G"],
                   choices=list(STAGE_CONFIGS.keys()))
    p.add_argument("--student",      default="qwen2.5-1.5b",
                   choices=["qwen2.5-1.5b", "phi3.5-mini"])
    p.add_argument("--datasets",     nargs="+",
                   default=["gsm8k", "math", "bbh", "arc_challenge"])
    p.add_argument("--traj-dir",     default="./data/trajectories")
    p.add_argument("--cache-dir",    default="./cache/huggingface")
    p.add_argument("--output-dir",   default="./outputs")
    p.add_argument("--max-steps",    type=int, default=1000)
    p.add_argument("--eval-every",   type=int, default=200)
    p.add_argument("--save-every",   type=int, default=500)
    p.add_argument("--batch-size",   type=int, default=4)
    p.add_argument("--grad-accum",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--t-max",        type=int, default=20)
    p.add_argument("--lambda-vf",    type=float, default=0.1)
    p.add_argument("--lambda-lya",   type=float, default=0.1)
    p.add_argument("--lambda-jac",   type=float, default=0.05)
    p.add_argument("--lambda-state", type=float, default=0.1)
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  LCLDD Training")
    print("=" * 60)
    print(f"  Device   : {device}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Stages   : {args.stages}")
    print(f"  Steps    : {args.max_steps} per stage")
    print(f"  Datasets : {args.datasets}")

    # ── Load student model ─────────────────────────────────────────────
    print("\n[1] Loading student base model...")
    bundle = load_student(args.student, cache_dir=args.cache_dir)

    student = LCLDDStudent(
        base_model=bundle.model,
        hidden_dim=bundle.hidden_dim,
        vocab_size=bundle.model.config.vocab_size,
        t_max=args.t_max,
        alpha=1.0,
        beta=0.1,
    ).to(device)

    total_params  = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params/1e6:.1f}M")

    # ── Resume from checkpoint ─────────────────────────────────────────
    if args.resume:
        print(f"\n[2] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        student.encoder.load_state_dict(ckpt["encoder"])
        student.thinking.load_state_dict(ckpt["thinking"])
        student.decoder.load_state_dict(ckpt["decoder"])
        student.halting.load_state_dict(ckpt["halting"])
        print(f"  Loaded stage {ckpt['stage']} step {ckpt['step']}")

    # ── Load trajectory data ───────────────────────────────────────────
    print("\n[2] Loading trajectory data...")
    dataloader, all_records = build_dataloaders(
        traj_dir=args.traj_dir,
        tokenizer=bundle.tokenizer,
        datasets=args.datasets,
        batch_size=args.batch_size,
    )

    # ── Build trainer ──────────────────────────────────────────────────
    trainer = Trainer(
        student=student,
        tokenizer=bundle.tokenizer,
        dataloader=dataloader,
        all_records=all_records,
        device=device,
        lr=args.lr,
        grad_accum=args.grad_accum,
        output_dir=args.output_dir,
        lambda_vf=args.lambda_vf,
        lambda_lya=args.lambda_lya,
        lambda_jac=args.lambda_jac,
        lambda_state=args.lambda_state,
    )

    # ── Run curriculum stages ──────────────────────────────────────────
    print("\n[3] Starting curriculum training...")
    for stage in args.stages:
        tracker = trainer.run_stage(
            stage=stage,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            save_every=args.save_every,
        )
        print(f"\n  Stage {stage} complete.")

    print("\n" + "=" * 60)
    print("  Training complete. Outputs saved to:", args.output_dir)
    print("  Plots  :", f"{args.output_dir}/plots/")
    print("  Metrics:", f"{args.output_dir}/metrics/")
    print("=" * 60)


if __name__ == "__main__":
    main()
