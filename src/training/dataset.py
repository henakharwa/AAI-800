"""
TrajectoryDataset — PyTorch Dataset that loads cached teacher trajectories.

Each sample returns:
  - input_ids / attention_mask  : tokenized question
  - label_ids                   : tokenized answer (first token for L_ans)
  - teacher_hidden              : (N, d_teacher) hidden states
  - teacher_motion              : (N-1, d_teacher) motion vectors
  - teacher_jacobian            : (seq, d_teacher) Jacobian
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
from pathlib import Path
from transformers import PreTrainedTokenizer


class TrajectoryDataset(Dataset):

    def __init__(
        self,
        records: List[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.records    = records
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]

        # Tokenize question
        enc = self.tokenizer(
            r["question"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Label: first token of the gold answer
        answer_enc = self.tokenizer(
            r["gold_answer"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        label = answer_enc["input_ids"][0, 0] if answer_enc["input_ids"].shape[1] > 0 \
                else torch.tensor(self.tokenizer.eos_token_id)

        return {
            "input_ids":        enc["input_ids"].squeeze(0),        # (seq,)
            "attention_mask":   enc["attention_mask"].squeeze(0),   # (seq,)
            "label":            label,                              # scalar
            "teacher_hidden":   r["hidden_states"],                 # (N, d_t)
            "teacher_motion":   r["motion_vectors"],                # (N-1, d_t)
            "teacher_jacobian": r["jacobian"],                      # (seq2, d_t)
            "gold_answer":      r["gold_answer"],
            "question":         r["question"],
            "dataset":          r["dataset"],
        }


def collate_fn(batch: List[dict]) -> dict:
    """Pad teacher tensors to the same N (number of steps) within a batch."""
    # Stack fixed-size tensors
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels         = torch.stack([b["label"]          for b in batch])

    # Pad teacher_hidden and teacher_motion to max steps in batch
    max_steps = max(b["teacher_hidden"].shape[0] for b in batch)
    max_steps_m1 = max(b["teacher_motion"].shape[0] for b in batch)
    d_t = batch[0]["teacher_hidden"].shape[-1]

    teacher_hidden_padded = torch.zeros(len(batch), max_steps, d_t)
    teacher_motion_padded = torch.zeros(len(batch), max_steps_m1, d_t)
    for i, b in enumerate(batch):
        n  = b["teacher_hidden"].shape[0]
        nm = b["teacher_motion"].shape[0]
        teacher_hidden_padded[i, :n, :]  = b["teacher_hidden"]
        teacher_motion_padded[i, :nm, :] = b["teacher_motion"]

    # Pad teacher_jacobian to max seq_len in batch
    max_jac_seq = max(b["teacher_jacobian"].shape[0] for b in batch)
    teacher_jac_padded = torch.zeros(len(batch), max_jac_seq, d_t)
    for i, b in enumerate(batch):
        s = b["teacher_jacobian"].shape[0]
        teacher_jac_padded[i, :s, :] = b["teacher_jacobian"]

    return {
        "input_ids":        input_ids,
        "attention_mask":   attention_mask,
        "labels":           labels,
        "teacher_hidden":   teacher_hidden_padded,
        "teacher_motion":   teacher_motion_padded,
        "teacher_jacobian": teacher_jac_padded,
        "gold_answers":     [b["gold_answer"] for b in batch],
        "questions":        [b["question"]    for b in batch],
        "datasets":         [b["dataset"]     for b in batch],
    }


def build_dataloaders(
    traj_dir: str,
    tokenizer: PreTrainedTokenizer,
    datasets: List[str],
    batch_size: int = 4,
    max_length: int = 256,
):
    """Load all cached trajectories and return a combined DataLoader."""
    import random
    all_records = []
    for name in datasets:
        path = Path(traj_dir) / f"{name}_train.pt"
        if path.exists():
            records = torch.load(path, weights_only=False)
            all_records.extend(records)
            print(f"  [Dataset] {name}: {len(records)} trajectories")
        else:
            print(f"  [Dataset] {name}: not found, skipping")

    random.shuffle(all_records)
    print(f"  [Dataset] Total: {len(all_records)} trajectories")

    dataset = TrajectoryDataset(all_records, tokenizer, max_length)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return loader, all_records
