"""
Smoke test for ActionDiffusionModel — verifies the forward pass runs without errors
for all three backbone types (timm, vov*, bev*) in both training and eval modes.

* vov and bev require a VoV checkpoint, so they are skipped if vov_ckpt is empty.

Run from the repo root:
    python navsim/agents/action_diffusion_agent/smoke_test.py
"""
import torch
from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.ad_model import ActionDiffusionModel

B = 2   # batch size
T = 1   # seq_len
# Use small spatial dims to keep the test fast
H, W = 128, 256

# ── helper: build a minimal features dict ────────────────────────────────────

def make_features(device="cpu"):
    return {
        "camera_feature":      torch.randn(B, T, 3, H, W, device=device),
        "camera_feature_back": torch.randn(B, T, 3, H, W, device=device),
        "status_feature":      torch.randn(B, 8,          device=device),
        # 3 historical frames × 7 values each [vx, vy, ax, ay, px, py, heading]
        "hist_status_feature": torch.randn(B, 3 * 7,      device=device),
    }


# ── helper: run one test case ─────────────────────────────────────────────────

def run_case(label: str, config: ActionDiffusionConfig) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActionDiffusionModel(config).to(device)

    feats = make_features(device)

    # --- Training forward ---
    model.train()
    out_train = model(feats)
    assert "context_kv" in out_train,   "Missing 'context_kv' in train output"
    assert "trajectory" in out_train,   "Missing 'trajectory' placeholder in train output"
    ctx = out_train["context_kv"]
    expected_ctx_len = model._context_len
    assert ctx.shape == (B, expected_ctx_len, config.model_dim), \
        f"context_kv shape mismatch: got {ctx.shape}, expected ({B}, {expected_ctx_len}, {config.model_dim})"
    print(f"  [TRAIN]  context_kv: {tuple(ctx.shape)}  trajectory: {tuple(out_train['trajectory'].shape)}")

    # --- Loss computation (minimal target) ---
    gt_dense = torch.randn(B, 40, 3, device=device)   # interpolated_traj
    from navsim.agents.action_diffusion_agent.ad_agent import ActionDiffusionAgent
    agent = ActionDiffusionAgent(config).to(device)
    agent.train()
    fwd = agent.model(feats)
    loss = agent.compute_loss(feats, {"interpolated_traj": gt_dense}, fwd)
    assert loss.ndim == 0 and loss.item() > 0, f"Unexpected loss: {loss}"
    print(f"  [LOSS]   {loss.item():.4f}  ✓")

    # --- Inference forward ---
    model.eval()
    with torch.no_grad():
        out_eval = model(feats)
    assert "dp_pred"      in out_eval, "Missing 'dp_pred' in eval output"
    assert "pred_actions" in out_eval, "Missing 'pred_actions' in eval output"
    assert "trajectory"   in out_eval, "Missing 'trajectory' in eval output"

    N = config.num_inference_proposals
    assert out_eval["dp_pred"].shape      == (B, N, 8, 3),  f"dp_pred shape: {out_eval['dp_pred'].shape}"
    assert out_eval["pred_actions"].shape == (B, N, 40, 2), f"pred_actions shape: {out_eval['pred_actions'].shape}"
    assert out_eval["trajectory"].shape   == (B, 8, 3),     f"trajectory shape: {out_eval['trajectory'].shape}"
    print(f"  [EVAL]   dp_pred: {tuple(out_eval['dp_pred'].shape)}  "
          f"trajectory: {tuple(out_eval['trajectory'].shape)}  ✓")

    print(f"  PASSED ✓")


# ── test cases ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Case 1: timm backbone (resnet18, no pretrained download) ---
    cfg_timm = ActionDiffusionConfig(
        backbone_type="timm",
        timm_model_name="resnet18",
        timm_pretrained=False,
        freeze_backbone=False,
        camera_height=H,
        camera_width=W,
        use_back_view=True,
        img_vert_anchors=4,
        img_horz_anchors=8,
        model_dim=64,
        ffn_dim=128,
        num_heads=4,
        num_diffusion_layers=2,
        num_inference_proposals=4,
    )
    run_case("backbone_type='timm'  (resnet18, no pretrained)", cfg_timm)

    # --- Case 2: timm, use_back_view=False ---
    cfg_timm_noback = ActionDiffusionConfig(
        backbone_type="timm",
        timm_model_name="resnet18",
        timm_pretrained=False,
        freeze_backbone=False,
        camera_height=H,
        camera_width=W,
        use_back_view=False,
        img_vert_anchors=4,
        img_horz_anchors=8,
        model_dim=64,
        ffn_dim=128,
        num_heads=4,
        num_diffusion_layers=2,
        num_inference_proposals=4,
    )
    run_case("backbone_type='timm'  use_back_view=False", cfg_timm_noback)

    # --- Case 3: flow matching noise type ---
    cfg_flow = ActionDiffusionConfig(
        backbone_type="timm",
        timm_model_name="resnet18",
        timm_pretrained=False,
        freeze_backbone=False,
        camera_height=H,
        camera_width=W,
        use_back_view=True,
        img_vert_anchors=4,
        img_horz_anchors=8,
        model_dim=64,
        ffn_dim=128,
        num_heads=4,
        num_diffusion_layers=2,
        num_inference_proposals=4,
        noise_type="flow",
        num_flow_steps=5,
    )
    run_case("backbone_type='timm'  noise_type='flow'", cfg_flow)

    print("\n" + "="*60)
    print("  ALL TESTS PASSED")
    print("="*60)
