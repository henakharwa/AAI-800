"""
Model loader for LCLDD.

Memory strategy for RTX 3070 (8.6 GB VRAM):
  - Teacher (7B/8B): 4-bit quantization via bitsandbytes (~4 GB on GPU)
  - Student (1.5B) : fp16 on GPU (~3 GB)
  - Student (3.8B) : fp16 on GPU (~7.6 GB) — load alone, not alongside teacher

Teachers are always frozen (no gradients). Students are trainable.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

load_dotenv()

# ── Model registry ────────────────────────────────────────────────────────────

TEACHER_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

STUDENT_MODELS = {
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}

# Hidden dimensions — used later by the Thinking Block
HIDDEN_DIMS = {
    "qwen2.5-7b":   3584,
    "llama3.1-8b":  4096,
    "phi3.5-mini":  3072,
    "qwen2.5-1.5b": 1536,
}


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """Holds a model and its tokenizer together."""
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    hidden_dim: int
    is_teacher: bool

    def __repr__(self):
        role = "Teacher" if self.is_teacher else "Student"
        params = sum(p.numel() for p in self.model.parameters()) / 1e9
        device = next(self.model.parameters()).device
        return (
            f"ModelBundle({role} | {self.name} | "
            f"{params:.2f}B params | device={device})"
        )


# ── 4-bit quantization config (for teachers) ──────────────────────────────────

def _bnb_4bit_config() -> BitsAndBytesConfig:
    """
    4-bit NF4 quantization with bfloat16 compute dtype.
    Reduces a 7B model from ~14 GB (fp16) to ~4 GB on GPU.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB extra
    )


# ── Teacher loader ─────────────────────────────────────────────────────────────

def load_teacher(
    name: str,
    cache_dir: str = "./cache/huggingface",
    quantize: bool = True,
    gpu_memory_gb: float = 8.0,
) -> ModelBundle:
    """
    Load a teacher LLM, frozen in evaluation mode.

    Args:
        name:           Key from TEACHER_MODELS (e.g. "qwen2.5-7b")
        cache_dir:      Where HuggingFace downloads are cached
        quantize:       If True, load in 4-bit NF4 (required for 8 GB GPUs)
        gpu_memory_gb:  How much GPU VRAM (GB) to allow for this model.
                        Remaining layers overflow to CPU RAM automatically.

    Returns:
        ModelBundle with frozen teacher model + tokenizer
    """
    if name not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher '{name}'. Choose from: {list(TEACHER_MODELS)}")

    hf_id = TEACHER_MODELS[name]
    hf_token = os.environ.get("HF_TOKEN")

    print(f"[Loader] Loading teacher: {name} ({hf_id})")
    print(f"[Loader] Quantization: {'4-bit NF4' if quantize else 'fp16'}")
    print(f"[Loader] GPU budget: {gpu_memory_gb} GB  (overflow -> CPU RAM)")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Explicit memory budget — prevents bitsandbytes CPU-offload error.
    # device_map="auto" with max_memory splits layers across GPU/CPU cleanly.
    max_memory = {0: f"{int(gpu_memory_gb)}GiB", "cpu": "48GiB"}

    load_kwargs = dict(
        cache_dir=cache_dir,
        token=hf_token,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
    )
    if quantize:
        load_kwargs["quantization_config"] = _bnb_4bit_config()
    else:
        load_kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    # Freeze all parameters — teacher is inference-only
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Loader] Teacher loaded. Params: {total/1e9:.2f}B total, {trainable} trainable (frozen)")

    return ModelBundle(
        name=name,
        model=model,
        tokenizer=tokenizer,
        hidden_dim=HIDDEN_DIMS[name],
        is_teacher=True,
    )


# ── Student loader ─────────────────────────────────────────────────────────────

def load_student(
    name: str,
    cache_dir: str = "./cache/huggingface",
    device: Optional[str] = None,
) -> ModelBundle:
    """
    Load a student SLM in fp16, trainable.

    Args:
        name:      Key from STUDENT_MODELS (e.g. "qwen2.5-1.5b")
        cache_dir: Where HuggingFace downloads are cached
        device:    Target device. Defaults to CUDA if available, else CPU.

    Returns:
        ModelBundle with trainable student model + tokenizer
    """
    if name not in STUDENT_MODELS:
        raise ValueError(f"Unknown student '{name}'. Choose from: {list(STUDENT_MODELS)}")

    hf_id = STUDENT_MODELS[name]
    hf_token = os.environ.get("HF_TOKEN")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Loader] Loading student: {name} ({hf_id})")
    print(f"[Loader] Device: {device} | dtype: fp16")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model — fp16, all parameters trainable
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        cache_dir=cache_dir,
        token=hf_token,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Loader] Student loaded. Params: {total/1e9:.2f}B total, {trainable/1e9:.2f}B trainable")

    return ModelBundle(
        name=name,
        model=model,
        tokenizer=tokenizer,
        hidden_dim=HIDDEN_DIMS[name],
        is_teacher=False,
    )
