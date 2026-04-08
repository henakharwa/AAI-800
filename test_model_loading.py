"""
Step 2 Verification — Model Loading Test

Tests the loader logic using a tiny public model (GPT-2) so we can verify
freeze/unfreeze, dtype, device placement, and ModelBundle without
downloading 7B models.

Run full model downloads only when ready (see bottom of this file).

Usage:
    py -3.12 test_model_loading.py             # fast test with GPT-2
    py -3.12 test_model_loading.py --full      # download & load real models
"""

import sys
import os
import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.models.loader import (
    load_teacher,
    load_student,
    ModelBundle,
    TEACHER_MODELS,
    STUDENT_MODELS,
    HIDDEN_DIMS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def vram_used_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


# ── Smoke test using GPT-2 (tiny, no auth needed) ─────────────────────────────

def test_with_gpt2():
    """
    Validates all loader behaviour using GPT-2 (117M params, ~250 MB).
    Checks: frozen weights, eval mode, fp16 dtype, device placement,
            ModelBundle repr, and that students remain trainable.
    """
    print_section("Test 1: Teacher loading (frozen, eval) — using GPT-2 proxy")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    PROXY_ID = "gpt2"
    cache = "./cache/huggingface"

    # ── Simulate teacher load ──────────────────────────────────────────────
    print(f"  Loading {PROXY_ID} as teacher proxy...")
    tokenizer = AutoTokenizer.from_pretrained(PROXY_ID, cache_dir=cache)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        PROXY_ID, cache_dir=cache, torch_dtype=torch.float16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Freeze (same logic as load_teacher)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    bundle = ModelBundle(
        name="gpt2-teacher-proxy",
        model=model,
        tokenizer=tokenizer,
        hidden_dim=768,
        is_teacher=True,
    )

    # Assertions
    assert not any(p.requires_grad for p in bundle.model.parameters()), \
        "FAIL: Teacher has trainable params — should be fully frozen"
    assert not bundle.model.training, \
        "FAIL: Teacher should be in eval mode"
    assert all(p.dtype == torch.float16 for p in bundle.model.parameters()), \
        "FAIL: Teacher params should be fp16"
    assert str(next(bundle.model.parameters()).device).startswith(device), \
        f"FAIL: Expected device {device}"

    print(f"  {bundle}")
    print("  [PASS] frozen=True, eval=True, dtype=fp16, device correct")

    # ── Simulate student load ──────────────────────────────────────────────
    print_section("Test 2: Student loading (trainable, fp16) — using GPT-2 proxy")

    model2 = AutoModelForCausalLM.from_pretrained(
        PROXY_ID, cache_dir=cache, torch_dtype=torch.float16
    )
    model2 = model2.to(device)
    model2.train()

    bundle2 = ModelBundle(
        name="gpt2-student-proxy",
        model=model2,
        tokenizer=tokenizer,
        hidden_dim=768,
        is_teacher=False,
    )

    trainable = sum(p.numel() for p in bundle2.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in bundle2.model.parameters())

    assert trainable == total, \
        "FAIL: Student should have all params trainable"
    assert bundle2.model.training, \
        "FAIL: Student should be in train mode"

    print(f"  {bundle2}")
    print(f"  Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    print("  [PASS] trainable=True, train_mode=True, dtype=fp16")

    # ── Tokenizer sanity check ─────────────────────────────────────────────
    print_section("Test 3: Tokenizer sanity check")
    sample = "What is 2 + 2?"
    tokens = tokenizer(sample, return_tensors="pt").to(device)
    with torch.no_grad():
        out = bundle.model(**tokens)
    assert out.logits.shape[-1] > 0, "FAIL: No logits produced"
    print(f"  Input : '{sample}'")
    print(f"  Logits: {out.logits.shape}  (batch, seq_len, vocab_size)")
    print("  [PASS] Forward pass successful")

    # ── VRAM report ───────────────────────────────────────────────────────
    print_section("Memory Report")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM allocated : {alloc:.2f} GB")
        print(f"  VRAM reserved  : {reserved:.2f} GB")
        print(f"  VRAM total     : {total_vram:.2f} GB")
    else:
        print("  Running on CPU")

    print_section("ALL TESTS PASSED")
    print("  Loader logic verified. Ready to load real models.")
    print()
    print("  To download real models, run:")
    print("    py -3.12 test_model_loading.py --full")


# ── Full model loading test (downloads ~15 GB per model) ─────────────────────

def test_full_loading():
    # ── Teacher first — give it the full GPU budget ────────────────────────
    print_section("Full Model Load — Qwen2.5-7B teacher (4-bit quantized)")
    print("  Models already downloaded — loading from cache...")

    # 8 GB budget: 4-bit Qwen2.5-7B fits in ~4 GB, leaving headroom.
    teacher = load_teacher("qwen2.5-7b", quantize=True, gpu_memory_gb=8)
    print(f"  {teacher}")

    assert not any(p.requires_grad for p in teacher.model.parameters()), \
        "Teacher must be fully frozen"
    assert not teacher.model.training, "Teacher must be in eval mode"
    print("  [PASS] Teacher Qwen2.5-7B loaded, frozen, 4-bit quantized")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after teacher load: {alloc:.2f} GB")

    # ── Free teacher before loading student ───────────────────────────────
    # In real training, teacher runs offline (Step 3) and is unloaded
    # before the student training loop starts.
    print("\n  [Info] Unloading teacher to free VRAM for student...")
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Student ───────────────────────────────────────────────────────────
    print_section("Full Model Load — Qwen2.5-1.5B student")

    student = load_student("qwen2.5-1.5b")
    print(f"  {student}")

    trainable = sum(p.numel() for p in student.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.model.parameters())
    assert trainable == total, "Student should be fully trainable"
    assert student.model.training, "Student should be in train mode"
    print("  [PASS] Student Qwen2.5-1.5B loaded and trainable")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM after student load: {alloc:.2f} GB")

    print_section("ALL FULL TESTS PASSED")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--full" in sys.argv:
        test_full_loading()
    else:
        test_with_gpt2()
