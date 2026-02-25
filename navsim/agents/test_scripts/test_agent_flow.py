#!/usr/bin/env python3
"""
test_agent_flow.py — Real-data train + inference smoke-test for NAVSIM agents.

Loads 1 real scene from the trainval split, uses the agent's own feature/target
builders to create a batch, then runs:
  1. Training step  (forward → compute_loss → backward + gradient norm check)
  2. Inference step (eval + no_grad forward, tensor shape + finiteness checks)

The agent and all its hyperparameters are configured through Hydra — exactly
the same mechanism as run_training.py.  Any config field can be overridden
from the command line using standard Hydra dot-notation syntax.

Requires:
    OPENSCENE_DATA_ROOT  env var pointing at the dataset root.

Usage
-----
    # Default: action_diffusion_agent with DDPM on CPU
    python test_agent_flow.py

    # Switch to flow matching
    python test_agent_flow.py agent.config.noise_type=flow

    # Flow matching on GPU
    python test_agent_flow.py agent.config.noise_type=flow device=cuda

    # Different agent entirely
    python test_agent_flow.py agent=ego_status_mlp_agent device=cuda

    # Any nested field works
    python test_agent_flow.py agent.config.num_inference_proposals=8 device=cuda
"""

import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

# ── Data paths ────────────────────────────────────────────────────────────────
_DATA_ROOT    = Path(os.environ["OPENSCENE_DATA_ROOT"])
_NAVSIM_LOGS  = _DATA_ROOT / "navsim_logs"  / "trainval"
_SENSOR_BLOBS = _DATA_ROOT / "sensor_blobs" / "trainval"

# ── Hydra config location ─────────────────────────────────────────────────────
# Uses its own minimal config (no train_test_split, no dataloader/trainer
# required fields) but shares the same agent/ config group via searchpath.
CONFIG_PATH = "../../planning/script/config/test"
CONFIG_NAME = "default_test"


# ── Scene utilities ───────────────────────────────────────────────────────────

def _find_valid_token(scene_loader: SceneLoader) -> str:
    """Return the first token whose front-camera images are all present on disk."""
    for token, frames in scene_loader.scene_frames_dicts.items():
        if all(
            (_SENSOR_BLOBS / frame["cams"]["CAM_F0"]["data_path"]).exists()
            for frame in frames
        ):
            return token
    raise RuntimeError(
        "No token found with complete sensor data — "
        "check that $OPENSCENE_DATA_ROOT/sensor_blobs/trainval is populated."
    )


def _load_one_scene(agent, device: str):
    """Load 1 real scene and build features/targets with the agent's own builders."""
    scene_filter = SceneFilter(max_scenes=200)
    scene_loader = SceneLoader(
        data_path=_NAVSIM_LOGS,
        original_sensor_path=_SENSOR_BLOBS,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    token = _find_valid_token(scene_loader)
    print(f"  Token   : {token}")

    scene       = scene_loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()

    features: dict = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))

    targets: dict = {}
    for builder in agent.get_target_builders():
        targets.update(builder.compute_targets(scene))

    features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
    targets  = {k: v.unsqueeze(0).to(device) for k, v in targets.items()}
    return features, targets


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # 'device' is not a field in default_training.yaml — accept it as an
    # optional CLI override (e.g. device=cuda) and fall back to cpu.
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Agent  : {cfg.agent._target_}")
    print(f"  Device : {device}")
    print(f"  Data   : {_NAVSIM_LOGS}")
    print(f"{'='*60}\n")

    # ── 1. Instantiate agent ──────────────────────────────────────────────────
    print("[1/4] Instantiating agent ...")
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.to(device)
    print(f"      {agent.__class__.__name__} ready.")

    # Print noise_type if present (ActionDiffusionAgent specific)
    noise_type = getattr(getattr(agent, "_config", None), "noise_type", None)
    if noise_type:
        print(f"      noise_type = {noise_type}")
    print()

    # ── 2. Load 1 real scene ──────────────────────────────────────────────────
    print("[2/4] Loading 1 scene from database ...")
    features, targets = _load_one_scene(agent, device)
    print(f"      Features : {list(features.keys())}")
    print(f"      Targets  : {list(targets.keys())}\n")

    # ── 3. Training step ──────────────────────────────────────────────────────
    print("[3/4] Training step (forward → loss → backward) ...")
    agent.train()
    predictions = agent.forward(features)
    loss = agent.compute_loss(features, targets, predictions)
    loss.backward()

    assert torch.isfinite(loss), f"Loss is NaN/Inf: {loss.item()}"

    total_norm = sum(
        p.grad.data.norm(2).item() ** 2
        for p in agent.parameters()
        if p.grad is not None
    ) ** 0.5

    print(f"      loss      = {loss.item():.6f}  ✓")
    print(f"      grad norm = {total_norm:.4f}  ✓\n")

    # ── 4. Inference step ─────────────────────────────────────────────────────
    print("[4/4] Inference step (eval + no_grad) ...")
    agent.eval()
    agent.zero_grad()
    with torch.no_grad():
        predictions = agent.forward(features)

    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            finite = val.isfinite().all()
            print(f"      predictions['{key}']"
                  f"  shape={tuple(val.shape)}"
                  f"  min={val.min():.3f}  max={val.max():.3f}"
                  f"  {'✓' if finite else '✗ NaN/Inf!'}")
        else:
            print(f"      predictions['{key}'] = {type(val)}")

    trajectory = predictions.get("trajectory")
    assert trajectory is not None, "'trajectory' key missing from predictions!"
    assert trajectory.shape[-2:] == (8, 3), \
        f"Expected trajectory shape (..., 8, 3), got {trajectory.shape}"
    assert trajectory.isfinite().all(), "Trajectory contains NaN/Inf!"

    print(f"\n{'='*60}")
    print(f"  ALL STEPS PASSED ✓")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
