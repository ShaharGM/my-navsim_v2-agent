"""
Generate perception-trajectory memory bank for nearest-neighbor context retrieval.

This script iterates over a NAVSIM dataset split and extracts:
  1. Perception vectors: mean-pooled backbone tokens [N, D]
  2. Ground-truth dense trajectories: [N, 40, 3]

The output is a torch checkpoint with structure:
  {
    "perception_vectors": Tensor[N, D],
    "gt_trajectories": Tensor[N, 40, 3],
  }

Usage:
  python gen_nn_memory.py \
    --backbone_type timm \
    --backbone_name resnet50 \
    --train_test_split navtrain \
    --output_path /path/to/memory_bank.pt
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.ad_features import (
    ActionDiffusionFeatureBuilder,
    ActionDiffusionTargetBuilder,
)
from navsim.agents.action_diffusion_agent.backbones import build_backbone
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader


logger = logging.getLogger(__name__)


def _default_common_config_path() -> Path:
    """Return NAVSIM common Hydra config path."""
    return Path(__file__).resolve().parents[2] / "planning" / "script" / "config" / "common"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate perception-trajectory memory bank for NN context retrieval."
    )
    parser.add_argument(
        "--backbone_type",
        type=str,
        default="timm",
        choices=["timm", "vov", "bev"],
        help="Backbone type: timm (ImageNet models), vov (VoVNet), or bev (VoV+BEV fusion).",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="resnet50",
        help="Timm model name when backbone_type='timm'. Ignored otherwise.",
    )
    parser.add_argument(
        "--vov_ckpt",
        type=str,
        default="",
        help="Path to VoV checkpoint when backbone_type='vov' or 'bev'.",
    )
    parser.add_argument(
        "--bev_ckpt",
        type=str,
        default="",
        help="Path to BEV checkpoint when backbone_type='bev'.",
    )
    parser.add_argument(
        "--train_test_split",
        type=str,
        default="navtrain",
        help="Split config name under config/common/train_test_split (e.g., navtrain, mini).",
    )
    parser.add_argument(
        "--scene_filter_name",
        type=str,
        default=None,
        help="Optional override for scene filter config name under train_test_split/scene_filter.",
    )
    parser.add_argument(
        "--config_common_path",
        type=str,
        default=str(_default_common_config_path()),
        help="Path to NAVSIM common Hydra config directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the memory bank checkpoint (e.g., /path/to/memory.pt).",
    )
    parser.add_argument(
        "--navsim_log_path",
        type=str,
        default=None,
        help="Path to NAVSIM logs. If None, uses NAVSIM_LOG_PATH env variable or defaults.",
    )
    parser.add_argument(
        "--sensor_path",
        type=str,
        default=None,
        help="Path to sensor blobs. If None, uses SENSOR_BLOBS_PATH env variable or defaults.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (for memory efficiency). Recommended: 1.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to process. If None, process all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run backbone on.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ActionDiffusionConfig:
    """Build ActionDiffusionConfig from command-line arguments."""
    return ActionDiffusionConfig(
        backbone_type=args.backbone_type,
        timm_model_name=args.backbone_name if args.backbone_type == "timm" else "resnet50",
        timm_pretrained=args.backbone_type == "timm",
        vov_ckpt=args.vov_ckpt,
        bev_ckpt=args.bev_ckpt,
    )


def get_scene_loader(
    args: argparse.Namespace,
    config: ActionDiffusionConfig,
) -> SceneLoader:
    """Build and return a SceneLoader using NAVSIM Hydra split/filter scheme."""
    split_cfg, scene_filter, data_path, sensor_path, scene_filter_name = _resolve_data_and_filter(args, config)

    if not data_path.exists():
        raise FileNotFoundError(
            f"NAVSIM log path not found: {data_path}. "
            "Please set --navsim_log_path or download the dataset."
        )

    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Sensor path not found: {sensor_path}. "
            "Please set --sensor_path or download the dataset."
        )

    logger.info(f"Loading NAVSIM split: {args.train_test_split}")
    logger.info(f"  Data split: {split_cfg.data_split}")
    logger.info(f"  Scene filter: {scene_filter_name}")
    logger.info(
        "  Scene filter params: history=%d future=%d interval=%d has_route=%s max_scenes=%s",
        scene_filter.num_history_frames,
        scene_filter.num_future_frames,
        scene_filter.frame_interval,
        scene_filter.has_route,
        scene_filter.max_scenes,
    )
    logger.info(f"  Log path: {data_path}")
    logger.info(f"  Sensor path: {sensor_path}")

    scene_loader = SceneLoader(
        original_sensor_path=sensor_path,
        data_path=data_path,
        scene_filter=scene_filter,
    )

    logger.info(f"Total scenes in {args.train_test_split}: {len(scene_loader)}")
    return scene_loader


def _resolve_data_and_filter(
    args: argparse.Namespace,
    config: ActionDiffusionConfig,
) -> Tuple[object, SceneFilter, Path, Path, str]:
    """Resolve split config, scene filter, and dataset paths from Hydra YAMLs."""
    common_cfg_path = Path(args.config_common_path)
    split_cfg_path = common_cfg_path / "train_test_split" / f"{args.train_test_split}.yaml"
    if not split_cfg_path.exists():
        raise FileNotFoundError(f"Split config not found: {split_cfg_path}")

    split_cfg = OmegaConf.load(split_cfg_path)
    scene_filter_name = args.scene_filter_name or _extract_scene_filter_name(split_cfg)

    scene_filter_cfg_path = common_cfg_path / "train_test_split" / "scene_filter" / f"{scene_filter_name}.yaml"
    if not scene_filter_cfg_path.exists():
        raise FileNotFoundError(f"Scene filter config not found: {scene_filter_cfg_path}")
    scene_filter_cfg = OmegaConf.load(scene_filter_cfg_path)
    scene_filter: SceneFilter = instantiate(scene_filter_cfg)

    required_future = int(config.internal_horizon)
    if scene_filter.num_future_frames < required_future:
        logger.warning(
            "Scene filter future horizon (%d) is smaller than required internal horizon (%d). "
            "Overriding num_future_frames to %d.",
            scene_filter.num_future_frames,
            required_future,
            required_future,
        )
        scene_filter.num_future_frames = required_future

    if args.max_scenes is not None:
        scene_filter.max_scenes = args.max_scenes

    default_paths_cfg_path = common_cfg_path / "default_dataset_paths.yaml"
    if not default_paths_cfg_path.exists():
        raise FileNotFoundError(f"Default dataset paths config not found: {default_paths_cfg_path}")

    dataset_paths_cfg = OmegaConf.load(default_paths_cfg_path)
    ctx = OmegaConf.create({"train_test_split": {"data_split": split_cfg.data_split}})
    resolved_paths = OmegaConf.merge(dataset_paths_cfg, ctx)
    OmegaConf.resolve(resolved_paths)

    data_path = Path(args.navsim_log_path) if args.navsim_log_path else Path(resolved_paths.navsim_log_path)
    sensor_path = Path(args.sensor_path) if args.sensor_path else Path(resolved_paths.original_sensor_path)
    return split_cfg, scene_filter, data_path, sensor_path, scene_filter_name


def _extract_scene_filter_name(split_cfg: object) -> str:
    """Extract `scene_filter` entry from split config defaults list."""
    defaults = split_cfg.get("defaults", [])
    for entry in defaults:
        if isinstance(entry, str) and entry.startswith("scene_filter:"):
            return entry.split(":", maxsplit=1)[1].strip()
        if OmegaConf.is_dict(entry) and "scene_filter" in entry:
            return str(entry["scene_filter"])
    raise ValueError("Split config must define defaults with a scene_filter entry.")


@torch.no_grad()
def extract_perception_trajectory_pairs(
    scene_loader: SceneLoader,
    config: ActionDiffusionConfig,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterate over scenes and extract (perception, trajectory) pairs.

    Returns:
        perception_bank: [N, D] float tensor
        trajectory_bank: [N, 40, 3] float tensor
    """
    # Build backbone and feature/target builders
    backbone = build_backbone(config)
    feature_builder = ActionDiffusionFeatureBuilder(config)
    target_builder = ActionDiffusionTargetBuilder(config)

    backbone.to(args.device)
    backbone.eval()

    perception_list: List[torch.Tensor] = []
    trajectory_list: List[torch.Tensor] = []

    num_scenes = len(scene_loader)
    if args.max_scenes is not None:
        num_scenes = min(args.max_scenes, num_scenes)

    logger.info(f"Processing {num_scenes} scenes...")

    for idx in tqdm(range(num_scenes), desc="Extracting perception-trajectory pairs"):
        try:
            token = scene_loader.tokens[idx]
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene_loader.get_agent_input_from_token(token)

            # Extract features
            features = feature_builder.compute_features(agent_input)
            targets = target_builder.compute_targets(scene)

            # Get dense GT trajectory (40, 3)
            gt_trajectory = targets["interpolated_traj"]  # (40, 3)

            # Extract perception by running backbone and mean-pooling
            with torch.no_grad():
                backbone_inputs = {
                    "camera_feature": features["camera_feature"].unsqueeze(0).to(args.device),
                }
                if "camera_feature_back" in features:
                    backbone_inputs["camera_feature_back"] = (
                        features["camera_feature_back"].unsqueeze(0).to(args.device)
                    )
                backbone_tokens = backbone(backbone_inputs)  # (1, N_tokens, C)
                perception_vec = backbone_tokens.mean(dim=1)  # (1, C)
                perception_vec = perception_vec.squeeze(0)  # (C,)

            perception_list.append(perception_vec.cpu())
            trajectory_list.append(gt_trajectory)

        except Exception as e:
            logger.warning(f"Error processing scene {idx} (token={token}): {e}")
            continue

    if not perception_list:
        raise RuntimeError("Failed to extract any valid scenes. Check dataset paths and configuration.")

    # Stack into tensors
    perception_bank = torch.stack(perception_list, dim=0)  # [N, D]
    trajectory_bank = torch.stack(trajectory_list, dim=0)  # [N, 40, 3]

    logger.info(f"Extracted {len(perception_list)} scenes")
    logger.info(f"Perception bank shape: {perception_bank.shape}")
    logger.info(f"Trajectory bank shape: {trajectory_bank.shape}")

    return perception_bank, trajectory_bank


def save_memory_bank(
    perception_bank: torch.Tensor,
    trajectory_bank: torch.Tensor,
    output_path: str,
    train_test_split: str,
) -> Path:
    """Save perception-trajectory memory bank to disk."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "perception_vectors": perception_bank,
        "gt_trajectories": trajectory_bank,
        "metadata": {
            "train_test_split": train_test_split,
            "num_samples": int(perception_bank.shape[0]),
        },
    }

    torch.save(payload, output_path)
    logger.info(f"Memory bank saved to: {output_path}")
    logger.info(f"  Perception vectors: {perception_bank.shape}")
    logger.info(f"  GT trajectories: {trajectory_bank.shape}")
    return output_path


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger.info("=" * 70)
    logger.info("Generating Perception-Trajectory Memory Bank for NN Context Retrieval")
    logger.info("=" * 70)
    logger.info(f"Backbone type: {args.backbone_type}")
    if args.backbone_type == "timm":
        logger.info(f"  Timm model: {args.backbone_name}")
    elif args.backbone_type == "vov":
        logger.info(f"  VoV checkpoint: {args.vov_ckpt}")
    elif args.backbone_type == "bev":
        logger.info(f"  VoV checkpoint: {args.vov_ckpt}")
        logger.info(f"  BEV checkpoint: {args.bev_ckpt}")
    logger.info(f"Dataset split: {args.train_test_split}")
    if args.scene_filter_name is not None:
        logger.info(f"Scene filter override: {args.scene_filter_name}")
    logger.info(f"Config common path: {args.config_common_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max scenes: {args.max_scenes or 'all'}")
    logger.info("=" * 70)

    # Build config and scene loader
    config = build_config(args)
    scene_loader = get_scene_loader(args, config)

    # Extract perception-trajectory pairs
    perception_bank, trajectory_bank = extract_perception_trajectory_pairs(
        scene_loader,
        config,
        args,
    )

    # Save memory bank
    saved_path = save_memory_bank(
        perception_bank,
        trajectory_bank,
        args.output_path,
        args.train_test_split,
    )

    logger.info(f"Saved pairs file: {saved_path}")
    logger.info("✓ Memory bank generation complete!")


if __name__ == "__main__":
    main()
