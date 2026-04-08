"""
EfficiencyProfiler — measures compute efficiency of the student model.

Metrics:
  - Wall-clock latency (ms) at varying batch sizes
  - Peak GPU memory (MB)
  - Estimated FLOPs per forward pass (approximate)
  - Steps used vs. accuracy trade-off (steps_used / accuracy)
  - Throughput (samples/sec)

Results saved to outputs/eval/efficiency_stage_X.json
"""

import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

from ..student.lcldd_student import LCLDDStudent


class EfficiencyProfiler:
    """
    Profiles the student model for latency, throughput, and memory.

    Usage:
        profiler = EfficiencyProfiler(student, tokenizer, device,
                                      save_dir="outputs/eval")
        results = profiler.profile(records, stage="A", use_halting=False)
        profiler.save(results, stage="A")
        profiler.print_report(results)
    """

    def __init__(
        self,
        student:   LCLDDStudent,
        tokenizer: PreTrainedTokenizer,
        device:    str = "cuda",
        save_dir:  str = "outputs/eval",
    ):
        self.student   = student
        self.tokenizer = tokenizer
        self.device    = device
        self.save_dir  = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def profile(
        self,
        records:     List[dict],
        stage:       str  = "A",
        use_halting: bool = False,
        n_warmup:    int  = 5,
        n_measure:   int  = 50,
        batch_sizes: List[int] = [1, 2, 4],
    ) -> dict:
        """
        Profile latency and memory across batch sizes.
        Uses n_warmup forward passes to warm up GPU, then n_measure passes.
        """
        self.student.eval()

        # Prepare a fixed-length batch for all experiments
        questions = [r["question"] for r in records[:max(batch_sizes)]]
        if len(questions) < max(batch_sizes):
            # Repeat if we don't have enough samples
            questions = (questions * (max(batch_sizes) // len(questions) + 1))

        results = {
            "stage":       stage,
            "use_halting": use_halting,
            "batch_results": {},
        }

        for bs in batch_sizes:
            print(f"  Profiling batch_size={bs}...")

            enc = self.tokenizer(
                questions[:bs],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(self.device)

            # Warm up
            for _ in range(n_warmup):
                _ = self.student(enc["input_ids"], enc["attention_mask"],
                                 use_halting=use_halting)

            # Synchronise GPU before timing
            if self.device == "cuda":
                torch.cuda.synchronize()

            # Measure memory before
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / 1e6

            # Timed runs
            latencies  = []
            steps_used = []

            for _ in range(n_measure):
                t0  = time.perf_counter()
                out = self.student(enc["input_ids"], enc["attention_mask"],
                                   use_halting=use_halting)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t1  = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                steps_used.append(out.steps_taken.float().mean().item())

            # Peak memory
            if self.device == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
            else:
                peak_mem_mb = 0.0

            lat_ms    = float(np.mean(latencies))
            lat_std   = float(np.std(latencies))
            lat_p95   = float(np.percentile(latencies, 95))
            throughput = bs / (lat_ms / 1000.0)   # samples/sec

            mean_steps = float(np.mean(steps_used))

            results["batch_results"][str(bs)] = {
                "batch_size":    bs,
                "latency_ms":    round(lat_ms, 2),
                "latency_std":   round(lat_std, 2),
                "latency_p95":   round(lat_p95, 2),
                "throughput":    round(throughput, 2),
                "peak_mem_mb":   round(peak_mem_mb, 1),
                "mean_steps":    round(mean_steps, 2),
            }

            print(f"    lat={lat_ms:.1f}ms (p95={lat_p95:.1f}ms) | "
                  f"throughput={throughput:.1f} samp/s | "
                  f"peak_mem={peak_mem_mb:.0f}MB | steps={mean_steps:.1f}")

        # Estimate FLOPs (approximate)
        results["flops_estimate"] = self._estimate_flops()

        self.student.train()
        return results

    def _estimate_flops(self) -> dict:
        """
        Rough FLOPs estimate based on model dimensions.
        Per ThinkingBlock step:
          - Cross-attention: 4 * seq_len * d^2  (Q,K,V,O projections)
          - MLP: 3 * d * (4d) = 12d^2           (gate, value, out projections)
          - Total per step ~ 16 * d^2
        Decoder: 2 * d * vocab_size
        """
        d     = self.student.hidden_dim
        vocab = self.student.vocab_size
        t     = self.student.t_max

        flops_per_step   = 16 * (d ** 2)
        flops_thinking   = t * flops_per_step
        flops_decoder    = 2 * d * vocab
        flops_total      = flops_thinking + flops_decoder

        return {
            "hidden_dim":        d,
            "vocab_size":        vocab,
            "t_max":             t,
            "flops_per_step":    flops_per_step,
            "flops_thinking":    flops_thinking,
            "flops_decoder":     flops_decoder,
            "flops_total_approx": flops_total,
            "gflops_approx":     round(flops_total / 1e9, 4),
        }

    def save(self, results: dict, stage: str):
        path = self.save_dir / f"efficiency_stage_{stage}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Efficiency] Saved -> {path}")

    def print_report(self, results: dict, stage: str = "?"):
        print(f"\n{'='*65}")
        print(f"  Efficiency Profile — Stage {stage}")
        print(f"{'='*65}")
        print(f"  {'BS':>4} {'Lat(ms)':>9} {'p95(ms)':>9} "
              f"{'Samp/s':>8} {'Mem(MB)':>9} {'Steps':>7}")
        print("  " + "-" * 63)
        for bs_str, m in results["batch_results"].items():
            print(f"  {m['batch_size']:>4} {m['latency_ms']:>9.1f} "
                  f"{m['latency_p95']:>9.1f} {m['throughput']:>8.1f} "
                  f"{m['peak_mem_mb']:>9.0f} {m['mean_steps']:>7.1f}")
        fe = results.get("flops_estimate", {})
        if fe:
            print(f"\n  Estimated GFLOPs/sample: {fe.get('gflops_approx', '?')}")
        print(f"{'='*65}")
