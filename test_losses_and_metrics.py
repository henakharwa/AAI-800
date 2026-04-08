"""
Step 5 Verification — Loss Functions, Metrics & Plotting

Tests all four loss functions against real student outputs and saved
teacher trajectories (from the 13-sample smoke test in Step 3).

Usage:
    py -3.12 test_losses_and_metrics.py
"""

import os, sys, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv; load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.models.loader import load_student
from src.student import LCLDDStudent
from src.training import LCLDDLoss, MetricsTracker, TrainingPlotter


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load student model ─────────────────────────────────────────────────
    print_section("Loading student model")
    bundle  = load_student("qwen2.5-1.5b")
    student = LCLDDStudent(
        base_model=bundle.model,
        hidden_dim=bundle.hidden_dim,
        vocab_size=bundle.model.config.vocab_size,
        t_max=5, alpha=1.0, beta=0.1,
    ).to(device)
    tokenizer = bundle.tokenizer
    print(f"  Student ready on {device}")

    # ── Load saved teacher trajectories ───────────────────────────────────
    print_section("Loading teacher trajectories (from Step 3 smoke test)")
    records = torch.load("./data/trajectories/gsm8k_train.pt", weights_only=False)
    print(f"  Loaded {len(records)} trajectories")
    r = records[0]
    print(f"  Sample: steps={r['n_steps']} | "
          f"hidden={tuple(r['hidden_states'].shape)} | "
          f"motion={tuple(r['motion_vectors'].shape)} | "
          f"jacobian={tuple(r['jacobian'].shape)}")

    # ── Run student forward pass ───────────────────────────────────────────
    print_section("Student forward pass")
    question = r["question"]
    inputs   = tokenizer(question, return_tensors="pt",
                         padding=True, truncation=True, max_length=256).to(device)
    output   = student(inputs["input_ids"], inputs["attention_mask"], use_halting=False)
    print(f"  Logits: {tuple(output.logits.shape)}")
    print(f"  Hidden states: {len(output.hidden_states)} steps")

    # ── Build loss inputs ──────────────────────────────────────────────────
    # Fake label: most confident token
    labels          = output.logits.argmax(dim=-1)             # (1,)
    h0_anchor       = output.hidden_states[0].to(device)       # (1, 1536)
    hidden_states   = [h.to(device) for h in output.hidden_states]

    # Teacher tensors from cache (add batch dim)
    teacher_motion  = r["motion_vectors"].unsqueeze(0).to(device)   # (1, N-1, d_t)
    teacher_hidden  = r["hidden_states"].unsqueeze(0).to(device)    # (1, N, d_t)
    teacher_jac     = r["jacobian"].unsqueeze(0).to(device)         # (1, seq, d_t)

    # Student Jacobian (simplified — gradient of h_T sum w.r.t. input embeds)
    embed = bundle.model.get_input_embeddings()
    with torch.no_grad():
        embeds = embed(inputs["input_ids"])
    embeds = embeds.detach().requires_grad_(True)
    base_out = bundle.model(inputs_embeds=embeds, output_hidden_states=True)
    h_T = base_out.hidden_states[-1][0, -1, :]
    grad = torch.autograd.grad(h_T.sum(), embeds)[0]
    student_jac = grad.detach()                                 # (1, seq, 1536)

    # ── Test each loss ─────────────────────────────────────────────────────
    loss_fn = LCLDDLoss(lambda_vf=0.1, lambda_lya=0.1, lambda_jac=0.05)

    print_section("Test 1: L_ans (Stage A)")
    out = loss_fn(logits=output.logits, labels=labels)
    assert out.lya is None and out.vf is None and out.jac is None
    print(f"  L_ans  = {out.ans.item():.4f}")
    print(f"  Total  = {out.total.item():.4f}")
    assert out.total.item() == out.ans.item(), "Stage A: total should equal L_ans"
    print("  [PASS]")

    print_section("Test 2: L_ans + L_lya (Stage E)")
    out = loss_fn(logits=output.logits, labels=labels,
                  hidden_states=hidden_states, h0_anchor=h0_anchor,
                  use_lya=True)
    assert out.lya is not None and out.lya.item() >= 0
    print(f"  L_ans  = {out.ans.item():.4f}")
    print(f"  L_lya  = {out.lya.item():.4f}")
    print(f"  Total  = {out.total.item():.4f}")
    print("  [PASS]")

    print_section("Test 3: L_ans + L_vf (Stage D)")
    out = loss_fn(logits=output.logits, labels=labels,
                  hidden_states=hidden_states,
                  teacher_motion=teacher_motion,
                  use_vf=True)
    assert out.vf is not None and out.vf.item() >= 0
    print(f"  L_ans  = {out.ans.item():.4f}")
    print(f"  L_vf   = {out.vf.item():.4f}")
    print(f"  Total  = {out.total.item():.4f}")
    print("  [PASS]")

    print_section("Test 4: Full loss (Stage F)")
    out = loss_fn(logits=output.logits, labels=labels,
                  hidden_states=hidden_states, h0_anchor=h0_anchor,
                  teacher_motion=teacher_motion,
                  teacher_hidden=teacher_hidden,
                  student_jacobian=student_jac,
                  teacher_jacobian=teacher_jac,
                  use_lya=True, use_vf=True, use_state=True, use_jac=True)
    print(f"  L_ans  = {out.ans.item():.4f}")
    print(f"  L_lya  = {out.lya.item():.4f}")
    print(f"  L_vf   = {out.vf.item():.4f}")
    print(f"  L_jac  = {out.jac.item():.4f}")
    print(f"  L_state= {out.state.item():.4f}")
    print(f"  Total  = {out.total.item():.4f}")
    assert out.total.item() > 0
    print("  [PASS]")

    print_section("Test 5: Backward pass (gradients flow)")
    out.total.backward()
    grads = [p.grad for p in student.thinking.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients reached ThinkingBlock"
    print(f"  ThinkingBlock params with gradients: {len(grads)}")
    print(f"  Max grad norm: {max(g.norm().item() for g in grads):.4f}")
    print("  [PASS]")

    # ── Metrics tracker ────────────────────────────────────────────────────
    print_section("Test 6: MetricsTracker")
    tracker = MetricsTracker("E_test", save_dir="outputs/metrics")

    # Simulate 100 training steps
    import math, random
    for step in range(1, 101):
        losses = {
            "loss_total": 2.5 * math.exp(-step/40) + random.gauss(0, 0.05),
            "loss_ans":   1.8 * math.exp(-step/40) + random.gauss(0, 0.03),
            "loss_lya":   0.7 * math.exp(-step/40) + random.gauss(0, 0.02),
        }
        tracker.log_step(
            step=step,
            losses=losses,
            lyapunov_energy=8000 * math.exp(-step/50) + random.gauss(0, 100),
            drift_magnitude=20   * math.exp(-step/60) + random.gauss(0, 0.5),
        )
        if step % 25 == 0:
            tracker.log_eval(step, {
                "gsm8k": 0.2 + 0.4 * (1 - math.exp(-step/60)) + random.gauss(0, 0.01),
                "math":  0.1 + 0.3 * (1 - math.exp(-step/60)) + random.gauss(0, 0.01),
            })

    tracker.save()
    tracker.summary()
    print("  [PASS]")

    # ── Plotter ────────────────────────────────────────────────────────────
    print_section("Test 7: TrainingPlotter — generating all plots")
    plotter = TrainingPlotter(tracker, save_dir="outputs/plots")
    plotter.plot_all()
    print("  [PASS]")

    print_section("ALL TESTS PASSED — Step 5 complete")
    print("  Loss functions  : L_ans, L_lya, L_vf, L_jac, L_state")
    print("  Backward pass   : gradients flow to ThinkingBlock")
    print("  MetricsTracker  : logs losses, energy, drift, accuracy")
    print("  TrainingPlotter : 6 plots generated in outputs/plots/")


if __name__ == "__main__":
    main()
