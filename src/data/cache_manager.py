"""
Trajectory cache — saves and loads teacher trajectories to/from disk.

Each dataset split is stored as a single .pt file:
    data/trajectories/<dataset>_<split>.pt

Each file contains a list of dicts (one per trajectory) with keys:
    sample_id, question, gold_answer, predicted_answer, dataset,
    hidden_states (Tensor), motion_vectors (Tensor), jacobian (Tensor), n_steps

A metadata JSON tracks what has been cached so far,
enabling resumable extraction (crash-safe incremental saving).
"""

import os
import json
import torch
from typing import List, Optional
from pathlib import Path

from .teacher_trajectories import Trajectory


class TrajectoryCache:
    """
    Manages saving and loading of teacher trajectories to disk.

    Usage:
        cache = TrajectoryCache("./data/trajectories")
        cache.save("gsm8k", "train", trajectories)
        loaded = cache.load("gsm8k", "train")
    """

    def __init__(self, cache_dir: str = "./data/trajectories"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "metadata.json"
        self._meta = self._load_meta()

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                return json.load(f)
        return {}

    def _save_meta(self):
        with open(self.meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

    def _key(self, dataset: str, split: str) -> str:
        return f"{dataset}_{split}"

    def _path(self, dataset: str, split: str) -> Path:
        return self.cache_dir / f"{dataset}_{split}.pt"

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, dataset: str, split: str, trajectories: List[Trajectory]):
        """Save trajectories to disk. Overwrites any existing file."""
        path = self._path(dataset, split)

        records = [
            {
                "sample_id":        t.sample_id,
                "question":         t.question,
                "gold_answer":      t.gold_answer,
                "predicted_answer": t.predicted_answer,
                "dataset":          t.dataset,
                "n_steps":          t.n_steps,
                "hidden_states":    t.hidden_states,    # (N, hidden_dim)
                "motion_vectors":   t.motion_vectors,   # (N-1, hidden_dim)
                "jacobian":         t.jacobian,         # (seq_len, hidden_dim)
            }
            for t in trajectories
        ]

        torch.save(records, path)

        self._meta[self._key(dataset, split)] = {
            "count": len(trajectories),
            "path": str(path),
            "avg_steps": sum(t.n_steps for t in trajectories) / max(len(trajectories), 1),
        }
        self._save_meta()
        print(f"[Cache] Saved {len(trajectories)} trajectories -> {path}")

    def append(self, dataset: str, split: str, new_trajectories: List[Trajectory]):
        """Append to existing cache (used for incremental/resumable extraction)."""
        existing = self.load(dataset, split) or []
        all_trajectories = existing + new_trajectories
        self.save(dataset, split, all_trajectories)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, dataset: str, split: str) -> Optional[List[dict]]:
        """
        Load trajectories from disk.
        Returns list of dicts, or None if not cached yet.
        """
        path = self._path(dataset, split)
        if not path.exists():
            return None
        records = torch.load(path, weights_only=False)
        print(f"[Cache] Loaded {len(records)} trajectories from {path}")
        return records

    def exists(self, dataset: str, split: str) -> bool:
        return self._path(dataset, split).exists()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary(self):
        """Print a summary of what is cached."""
        print("\n[Cache] Trajectory cache summary:")
        print(f"  Directory: {self.cache_dir}")
        if not self._meta:
            print("  (empty)")
            return
        for key, info in self._meta.items():
            print(f"  {key:<30} {info['count']:>5} trajectories  "
                  f"avg_steps={info['avg_steps']:.1f}")
