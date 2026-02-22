#!/usr/bin/env python3
"""
test_agent_flow.py — Real-data train + inference smoke-test for NAVSIM agents.

Loads 1 real scene from the trainval split, uses the agent's own feature/target
builders to create a batch, then runs:
  1. Training step  (forward → compute_loss → backward)
  2. Inference step (forward under torch.no_grad)

The agent is fully generic — no per-agent batch factories needed.

Requires:
    OPENSCENE_DATA_ROOT  env var pointing at the dataset root.

Usage
-----
    python test_agent_flow.py --agent action_diffusion_agent
    python test_agent_flow.py --agent action_diffusion_agent --device cuda
"""

import argparse
import os
from pathlib import Path

import navsim as _navsim_pkg
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

# ── Derived paths (no hardcoded relative tricks) ──────────────────────────────
_DATA_ROOT     = Path(os.environ["OPENSCENE_DATA_ROOT"])
_NAVSIM_LOGS   = _DATA_ROOT / "navsim_logs"  / "trainval"
_SENSOR_BLOBS  = _DATA_ROOT / "sensor_blobs" / "trainval"
# Agent YAMLs live inside the installed navsim package itself.
_AGENT_CFG_DIR = Path(_navsim_pkg.__file__).parent / "planning/script/config/common/agent"


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="NAVSIM agent smoke-test (real data)")
    parser.add_argument("--agent",  required=True,
                        help="Agent YAML name, e.g. action_diffusion_agent")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


# ── Agent loading ─────────────────────────────────────────────────────────────
def load_agent(agent_name: str, device: str):
    yaml_path = _AGENT_CFG_DIR / f"{agent_name}.yaml"
    assert yaml_path.exists(), f"Agent YAML not found: {yaml_path}"
    cfg = OmegaConf.load(yaml_path)
    agent = instantiate(cfg)
    agent.initialize()
    return agent.to(device)


# ── Token selection ───────────────────────────────────────────────────────────
def find_valid_token(scene_loader: SceneLoader) -> str:
    """
    Return the first token whose sensor images are all present on disk.

    scene_loader.scene_frames_dicts maps token → list of frame dicts.
    Each frame dict has frame["cams"]["CAM_F0"]["data_path"] — the relative
    path from sensor_blobs_path to the front-camera image.  We check this
    for every frame in the scene before committing to that token.
    """
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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_one_scene(agent, device: str):
    """Load 1 real scene and build features/targets with the agent's own builders."""
    scene_filter = SceneFilter(max_scenes=200)  # scan enough to find a valid token
    scene_loader = SceneLoader(
        data_path=_NAVSIM_LOGS,
        original_sensor_path=_SENSOR_BLOBS,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    token = find_valid_token(scene_loader)
    print(f"  Token  : {token}")

    scene       = scene_loader.get_scene_from_token(token)
    agent_input = scene.get_agent_input()

    features: dict = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))

    targets: dict = {}
    for builder in agent.get_target_builders():
        targets.update(builder.compute_targets(scene))

    # Add batch dimension and move to device
    features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}
    targets  = {k: v.unsqueeze(0).to(device) for k, v in targets.items()}
    return features, targets


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = args.device

    print(f"\n{'='*60}")
    print(f"  Agent  : {args.agent}")
    print(f"  Data   : {_NAVSIM_LOGS}")
    print(f"  Sensors: {_SENSOR_BLOBS}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # 1. Build agent
    print("[1/4] Loading agent from YAML ...")
    agent = load_agent(args.agent, device)
    print(f"      {agent.__class__.__name__} ready.\n")

    # 2. Load 1 real scene
    print("[2/4] Loading 1 scene from database ...")
    features, targets = load_one_scene(agent, device)
    print(f"      Features: {list(features.keys())}")
    print(f"      Targets : {list(targets.keys())}\n")

    # 3. Training step
    print("[3/4] Training step ...")
    agent.train()
    predictions = agent.forward(features)
    loss = agent.compute_loss(features, targets, predictions)
    loss.backward()
    print(f"      loss = {loss.item():.4f}  ✓\n")

    # 4. Inference step
    print("[4/4] Inference step ...")
    agent.eval()
    with torch.no_grad():
        predictions = agent.forward(features)

    for key, val in predictions.items():
        shape = val.shape if isinstance(val, torch.Tensor) else type(val)
        print(f"      predictions['{key}'] shape: {shape}")

    print("\n  ALL STEPS PASSED ✓\n")


if __name__ == "__main__":
    main()
