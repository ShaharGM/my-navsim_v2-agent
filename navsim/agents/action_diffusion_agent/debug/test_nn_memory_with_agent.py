"""
Visualization test for NN memory-bank retrieval using real agent backbone.

This script:
1. Loads random scenes from a NAVSIM split
2. Extracts backbone embeddings via the agent model
3. Queries the memory bank for nearest-neighbor GT trajectories
4. Plots the retrieved trajectory alongside ego state for inspection

This validates end-to-end integration: data → backbone → memory → retrieval.

Usage:
  python test_nn_memory_with_agent.py \
    --bank_path ../bev_navmini_perc_traj_pairs.pt \
    --backbone_type bev \
    --vov_ckpt /path/to/dd3d_det_final.pth \
    --bev_ckpt /path/to/gtrs_dp.ckpt \
    --train_test_split navmini \
    --num_scenes 3 \
    --output_dir ./nn_memory_viz
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from hydra.utils import instantiate
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import OmegaConf

from navsim.agents.action_diffusion_agent.ad_config import ActionDiffusionConfig
from navsim.agents.action_diffusion_agent.ad_features import ActionDiffusionFeatureBuilder
from navsim.agents.action_diffusion_agent.ad_model import ActionDiffusionModel
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_bev_ax, configure_ax

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize NN memory-bank retrieval using agent backbone."
    )
    parser.add_argument(
        "--bank_path",
        type=str,
        required=True,
        help="Path to memory bank .pt file.",
    )
    parser.add_argument(
        "--backbone_type",
        type=str,
        choices=["timm", "vov", "bev"],
        default="bev",
        help="Backbone type.",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="resnet50",
        help="Timm model name (if backbone_type='timm').",
    )
    parser.add_argument(
        "--vov_ckpt",
        type=str,
        default="",
        help="VoV checkpoint path.",
    )
    parser.add_argument(
        "--bev_ckpt",
        type=str,
        default="",
        help="BEV checkpoint path.",
    )
    parser.add_argument(
        "--train_test_split",
        type=str,
        default="navmini",
        help="NAVSIM split config name.",
    )
    parser.add_argument(
        "--config_common_path",
        type=str,
        default=None,
        help="Path to NAVSIM common config. If None, uses default.",
    )
    parser.add_argument(
        "--navsim_log_path",
        type=str,
        default=None,
        help="NAVSIM logs path. If None, uses defaults.",
    )
    parser.add_argument(
        "--sensor_path",
        type=str,
        default=None,
        help="Sensor blobs path. If None, uses defaults.",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=3,
        help="Number of random scenes to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for scene selection. If None, uses time-based randomness for different scenes each run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./nn_memory_viz",
        help="Directory to save visualization plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    parser.add_argument(
        "--use_faiss_retrieval",
        action="store_true",
        help="Enable FAISS-based retrieval flow (otherwise uses brute-force retrieval).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization plots. If disabled, only logs trajectory distance metrics.",
    )
    return parser.parse_args()


def _get_default_common_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "planning" / "script" / "config" / "common"


def _build_scene_loader(
    train_test_split: str,
    config_common_path: Optional[Path],
    navsim_log_path: Optional[Path],
    sensor_path: Optional[Path],
) -> SceneLoader:
    if config_common_path is None:
        config_common_path = _get_default_common_config_path()
    else:
        config_common_path = Path(config_common_path)

    split_cfg_path = config_common_path / "train_test_split" / f"{train_test_split}.yaml"
    split_cfg = OmegaConf.load(split_cfg_path)

    defaults = split_cfg.get("defaults", [])
    scene_filter_name = None
    for entry in defaults:
        if isinstance(entry, str) and entry.startswith("scene_filter:"):
            scene_filter_name = entry.split(":", maxsplit=1)[1].strip()
            break
        if OmegaConf.is_dict(entry) and "scene_filter" in entry:
            scene_filter_name = str(entry["scene_filter"])
            break

    scene_filter_cfg_path = config_common_path / "train_test_split" / "scene_filter" / f"{scene_filter_name}.yaml"
    scene_filter_cfg = OmegaConf.load(scene_filter_cfg_path)
    scene_filter: SceneFilter = instantiate(scene_filter_cfg)

    dataset_paths_cfg_path = config_common_path / "default_dataset_paths.yaml"
    dataset_paths_cfg = OmegaConf.load(dataset_paths_cfg_path)
    ctx = OmegaConf.create({"train_test_split": {"data_split": split_cfg.data_split}})
    resolved_paths = OmegaConf.merge(dataset_paths_cfg, ctx)
    OmegaConf.resolve(resolved_paths)

    if navsim_log_path is None:
        navsim_log_path = Path(resolved_paths.navsim_log_path)
    else:
        navsim_log_path = Path(navsim_log_path)

    if sensor_path is None:
        sensor_path = Path(resolved_paths.original_sensor_path)
    else:
        sensor_path = Path(sensor_path)

    all_frames = list(range(scene_filter.num_history_frames))
    sensor_config = SensorConfig.build_all_sensors(include=all_frames)
    sensor_config.lidar_pc = []
    sensor_config.cam_l1 = []
    sensor_config.cam_r1 = []
    sensor_config.cam_l2 = []
    sensor_config.cam_r2 = []
    sensor_config.cam_b0 = []

    return SceneLoader(
        original_sensor_path=sensor_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Set seed if provided, otherwise use random (system entropy)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Build config and model
    cfg = ActionDiffusionConfig(
        backbone_type=args.backbone_type,
        timm_model_name=args.backbone_name,
        timm_pretrained=args.backbone_type == "timm",
        vov_ckpt=args.vov_ckpt,
        bev_ckpt=args.bev_ckpt,
        use_nn_trajectory_context=True,
        nn_memory_path=args.bank_path,
        use_faiss_retrieval=args.use_faiss_retrieval,
    )

    logger.info("Building model...")
    model = ActionDiffusionModel(cfg)
    model.to(args.device)
    model.eval()

    logger.info("Building scene loader...")
    scene_loader = _build_scene_loader(
        args.train_test_split,
        Path(args.config_common_path) if args.config_common_path else None,
        Path(args.navsim_log_path) if args.navsim_log_path else None,
        Path(args.sensor_path) if args.sensor_path else None,
    )

    feature_builder = ActionDiffusionFeatureBuilder(cfg)

    n_scenes = len(scene_loader)
    sample_indices = np.random.choice(n_scenes, size=min(args.num_scenes, n_scenes), replace=False)

    logger.info(f"Sampling {len(sample_indices)} scenes from {n_scenes} available...")

    _k = 10
    trajectory_distances = []
    query_times = []

    with torch.no_grad():
        for sample_id, scene_idx in tqdm(enumerate(sample_indices), total=len(sample_indices), desc="Processing scenes"):
            token = scene_loader.tokens[int(scene_idx)]
            # logger.info(f"\n[{sample_id+1}/{len(sample_indices)}] Scene {scene_idx}: {token}")

            try:
                scene = scene_loader.get_scene_from_token(token)
                agent_input = scene_loader.get_agent_input_from_token(token)
                features = feature_builder.compute_features(agent_input)

                # Move features to device
                features_device = {
                    k: v.unsqueeze(0).to(args.device) if not isinstance(v, list) else v
                    for k, v in features.items()
                }

                # Forward through backbone to get perception vector
                backbone_tokens = model.backbone(features_device)  # (1, N_tokens, C)
                perception_vec = backbone_tokens.mean(dim=1)  # (1, C)

                # Query memory
                query_start = time.time()
                nn_traj_topk = model._nn_memory.query(perception_vec, k=_k)  # (1, k, 40, 3)
                query_time = time.time() - query_start
                query_times.append(query_time)
                nn_traj_topk_np = nn_traj_topk[0].cpu().numpy()  # (k, 40, 3)
                nn_traj_top1_np = nn_traj_topk_np[0]  # (40, 3)

                # Convert NN trajectory to Trajectory object for visualization
                # NN trajectories are 40 timesteps with 0.1s interval = 4s horizon
                traj_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
                nn_trajectories = [
                    Trajectory(poses=nn_traj_topk_np[i], trajectory_sampling=traj_sampling)
                    for i in range(_k)
                ]

                # Get human GT trajectory with matching 4s horizon (8 waypoints at 0.5s interval)
                human_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)

                # Calculate distance between trajectories
                nn_traj_sampled = nn_traj_top1_np[4::5]
                human_traj_poses = human_trajectory.poses
                
                # Calculate mean Euclidean distance between corresponding waypoints
                if len(nn_traj_sampled) >= len(human_traj_poses):
                    distances = np.linalg.norm(
                        nn_traj_sampled[:len(human_traj_poses), :2] - human_traj_poses[:, :2],
                        axis=1
                    )
                    mean_distance = np.mean(distances)
                    trajectory_distances.append(mean_distance)

                if args.visualize:
                    # Create BEV plot
                    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                    
                    # Render base scene (map, agents, etc.)
                    frame_idx = scene.scene_metadata.num_history_frames - 1
                    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

                    # Overlay human GT trajectory (green)
                    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])

                    # Overlay top-k NN retrieved trajectories
                    for i, nn_trajectory in enumerate(nn_trajectories):
                        nn_config = TRAJECTORY_CONFIG["agent"].copy()
                        nn_config["alpha"] = max(0.35, 0.85 - 0.12 * i)
                        add_trajectory_to_bev_ax(ax, nn_trajectory, nn_config)

                    # Configure axes
                    configure_bev_ax(ax)
                    configure_ax(ax)
                    
                    ax.set_title(
                        f"Scene {scene_idx}: NN Memory-Bank Retrieval (Top-{_k})\n"
                        f"Token: {token}\n"
                        f"Green=Human GT | Red shades=NN Retrieved Top-{_k}",
                        fontsize=14
                    )

                    plot_path = output_dir / f"scene_{scene_idx:04d}_nn_retrieval.png"
                    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                    logger.info(f"  Saved: {plot_path}")
                    plt.close(fig)

            except Exception as e:
                logger.error(f"  Error processing scene {scene_idx}: {e}", exc_info=True)

    logger.info("\n" + "="*60)
    if trajectory_distances:
        mean_all_distances = np.mean(trajectory_distances)
        logger.info(f"Mean trajectory distance across all scenes: {mean_all_distances:.4f}m")
    
    if query_times:
        mean_query_time = np.mean(query_times)
        logger.info(f"Mean query time: {mean_query_time:.4f}s")
    
    if args.visualize:
        logger.info(f"Plots saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
