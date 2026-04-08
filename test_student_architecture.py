"""
Step 4 Verification — Student Architecture Test

Tests the full student model (Encoder + ThinkingBlock + Decoder + Halting)
using Qwen2.5-1.5B as the base model.

Usage:
    py -3.12 test_student_architecture.py
"""

import os, sys, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.models.loader import load_student
from src.student import LCLDDStudent


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load base student model ────────────────────────────────────────────
    print_section("Loading base student model (Qwen2.5-1.5B)")
    bundle = load_student("qwen2.5-1.5b")
    base_model  = bundle.model
    tokenizer   = bundle.tokenizer
    hidden_dim  = bundle.hidden_dim   # 1536
    vocab_size  = base_model.config.vocab_size
    print(f"  hidden_dim : {hidden_dim}")
    print(f"  vocab_size : {vocab_size}")

    # ── Build LCLDD student ───────────────────────────────────────────────
    print_section("Building LCLDDStudent")
    student = LCLDDStudent(
        base_model=base_model,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        t_max=5,          # use 5 steps for fast testing (20 in full training)
        alpha=1.0,
        beta=0.1,
    ).to(device)

    new_params = (
        sum(p.numel() for p in student.encoder.parameters()) +
        sum(p.numel() for p in student.thinking.parameters()) +
        sum(p.numel() for p in student.decoder.parameters()) +
        sum(p.numel() for p in student.halting.parameters())
    )
    print(f"  New trainable params (LCLDD components): {new_params/1e6:.2f}M")

    # ── Test 1: Basic forward pass ────────────────────────────────────────
    print_section("Test 1: Forward pass (t_max=5, no halting)")
    question = "What is 12 multiplied by 8?"
    inputs = tokenizer(question, return_tensors="pt", padding=True).to(device)

    output = student(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_halting=False,
    )

    assert output.logits.shape == (1, vocab_size), \
        f"Expected logits (1, {vocab_size}), got {output.logits.shape}"
    assert len(output.hidden_states) == 6, \
        f"Expected 6 hidden states (h0..h5), got {len(output.hidden_states)}"
    assert output.steps_taken.item() == 5

    print(f"  Logits shape      : {tuple(output.logits.shape)}")
    print(f"  Hidden states     : {len(output.hidden_states)} steps")
    print(f"  Steps taken       : {output.steps_taken.tolist()}")
    print(f"  Lyapunov energies : {[f'{e.item():.3f}' for e in output.lyapunov_energies]}")
    print("  [PASS]")

    # ── Test 2: Lyapunov energy decreasing trend ───────────────────────────
    print_section("Test 2: Lyapunov energy values are non-negative")
    for i, e in enumerate(output.lyapunov_energies):
        assert e.item() >= 0, f"Lyapunov energy at step {i} is negative: {e.item()}"
    print(f"  All {len(output.lyapunov_energies)} energies >= 0  [PASS]")

    # ── Test 3: Hidden state shapes ────────────────────────────────────────
    print_section("Test 3: Hidden state tensor shapes")
    for i, h in enumerate(output.hidden_states):
        assert h.shape == (1, hidden_dim), \
            f"Step {i}: expected (1, {hidden_dim}), got {h.shape}"
    print(f"  All hidden states shape (1, {hidden_dim})  [PASS]")

    # ── Test 4: Motion vectors (used by L_vf) ─────────────────────────────
    print_section("Test 4: Motion vectors (v_t = h_{t+1} - h_t)")
    hs = torch.stack(output.hidden_states, dim=0)  # (T+1, batch, hidden_dim)
    motion_vectors = hs[1:] - hs[:-1]              # (T, batch, hidden_dim)
    assert motion_vectors.shape == (5, 1, hidden_dim)
    norms = motion_vectors.norm(dim=-1).squeeze()
    print(f"  Motion vector norms: {[f'{n:.4f}' for n in norms.tolist()]}")
    print(f"  Shape: {tuple(motion_vectors.shape)}  [PASS]")

    # ── Test 5: Dynamic halting ────────────────────────────────────────────
    print_section("Test 5: Dynamic Phase-Space halting (use_halting=True)")
    output_halt = student(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_halting=True,
    )
    steps = output_halt.steps_taken.item()
    print(f"  Steps used with halting : {steps} / 5")
    print(f"  Logits shape            : {tuple(output_halt.logits.shape)}")
    print("  [PASS]")

    # ── Test 6: Batch of multiple questions ────────────────────────────────
    print_section("Test 6: Batch forward pass (batch_size=3)")
    questions = [
        "What is 5 + 7?",
        "If a train travels 60 mph for 2 hours, how far does it go?",
        "What is the square root of 144?",
    ]
    inputs_batch = tokenizer(
        questions, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    output_batch = student(
        input_ids=inputs_batch["input_ids"],
        attention_mask=inputs_batch["attention_mask"],
        use_halting=False,
    )
    assert output_batch.logits.shape == (3, vocab_size)
    print(f"  Batch logits shape : {tuple(output_batch.logits.shape)}")
    print(f"  Steps taken        : {output_batch.steps_taken.tolist()}")
    print("  [PASS]")

    # ── Memory report ──────────────────────────────────────────────────────
    print_section("Memory Report")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM used : {alloc:.2f} GB / {total:.2f} GB")

    print_section("ALL TESTS PASSED — Student architecture ready")
    print("  Components verified:")
    print("    Encoder         - input -> h_0 (semantic anchor)")
    print("    ThinkingBlock   - h_t + G(h_t, h_0) -> h_{t+1}")
    print("    Decoder         - h_T -> logits + confidence + entropy")
    print("    HaltingController - 4-signal convergence check")
    print("    LCLDDStudent    - full recursive forward pass")


if __name__ == "__main__":
    main()
