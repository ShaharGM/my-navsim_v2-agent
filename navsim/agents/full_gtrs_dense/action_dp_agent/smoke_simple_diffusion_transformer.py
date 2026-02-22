import torch

from navsim.agents.full_gtrs_dense.action_dp_agent.dp_model import (
    ACTION_DIM,
    INTERNAL_HORIZON,
    SimpleDiffusionTransformer,
)


def run_smoke() -> None:
    torch.manual_seed(0)

    batch = 2
    d_model = 64
    nhead = 4
    d_ffn = 128
    layers = 2
    obs_len = 6
    d_cond = 32

    sample = torch.randn(batch, INTERNAL_HORIZON, ACTION_DIM)
    cond = torch.randn(batch, obs_len, d_cond)
    # timestep = torch.tensor(5, dtype=torch.long)
    timesteps = torch.randint(
            0, 1000, (batch,), device=sample.device
        ).long()
    

    print("Building SimpleDiffusionTransformer ...")
    try:
        model = SimpleDiffusionTransformer(
            d_model=d_model,
            nhead=nhead,
            d_ffn=d_ffn,
            d_cond=d_cond,
            dp_nlayers=layers,
            input_dim=ACTION_DIM * INTERNAL_HORIZON,
            obs_len=obs_len,
        )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"Init failed: {exc}")
        return

    print("Running forward + backward ...")
    out = model(sample, timesteps, cond)
    assert out.shape == sample.shape, f"expected {sample.shape}, got {out.shape}"
    out.mean().backward()

    print("Smoke test passed: output shape matches input and gradients flow.")


if __name__ == "__main__":
    run_smoke()
