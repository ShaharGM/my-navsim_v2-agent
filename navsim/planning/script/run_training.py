import logging
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from torch.utils.data import Subset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    train_shuffle = True

    if cfg.get("debug_overfit", False):
        print("\n" + "="*50)
        print("🚨 DEBUG MODE: OVERFITTING ON 4 DIVERSE SCENES 🚨")
        
        # Find exactly: 1 left, 1 right, 2 straight (stop as soon as we have them)
        import numpy as np
        cmd_names = ["left", "straight", "right", "unclear"]
        scenes_by_command = {0: None, 1: [], 2: None}  # left, straight (need 2), right
        
        for i, sample in enumerate(train_data):
            driving_command = sample[0]['status_feature'][:4]
            cmd_type = int(np.argmax(driving_command.numpy()))
            
            if cmd_type == 0 and scenes_by_command[0] is None:  # left
                scenes_by_command[0] = i
            elif cmd_type == 2 and scenes_by_command[2] is None:  # right
                scenes_by_command[2] = i
            elif cmd_type == 1 and len(scenes_by_command[1]) < 2:  # straight
                scenes_by_command[1].append(i)
            
            # Stop once we have all 4 scenes
            if (scenes_by_command[0] is not None and 
                scenes_by_command[2] is not None and 
                len(scenes_by_command[1]) == 2):
                break
        
        # Build subset indices
        # subset_indices = [0, 50, 100, 150]
        subset_indices = []
        if scenes_by_command[0] is not None:
            subset_indices.append(scenes_by_command[0])
        if scenes_by_command[2] is not None:
            subset_indices.append(scenes_by_command[2])
        subset_indices.extend(scenes_by_command[1])
        
        # Verify selection
        print(f"\n✓ Selected scenes:")
        for idx in subset_indices:
            sample = train_data[idx]
            cmd = sample[0]['status_feature'][:4]
            cmd_type = int(np.argmax(cmd.numpy()))
            print(f"   Scene {idx}: {cmd_names[cmd_type]}")
        
        assert len(subset_indices) == 4, f"Expected 4 scenes, got {len(subset_indices)}"
        train_data = Subset(train_data, indices=subset_indices)
        
        # B. FORCE Validation to be identical to Training
        val_data = train_data 
        
        with open_dict(cfg):
            # C. Dataloader Params
            cfg.dataloader.params.batch_size = 4
            cfg.dataloader.params.num_workers = 0
            cfg.dataloader.params.prefetch_factor = None
            
            # D. Trainer Params - ENABLE VALIDATION EVERY EPOCH
            cfg.trainer.params.limit_train_batches = 1.0 
            cfg.trainer.params.limit_val_batches = 1.0      # Run full validation (which is just 4 samples now)
            cfg.trainer.params.check_val_every_n_epoch = 1  # Check every single epoch
            cfg.trainer.params.num_sanity_val_steps = 0     # Skip the pre-train sanity check to save time
            cfg.trainer.params.log_every_n_steps = 1

        # E. Disable Shuffle so batches are consistent
        train_shuffle = False
        
        print(f"🎯 Training & Validating on same {len(train_data)} samples.")
        print("="*50 + "\n")
    # -------------------------------------------------------------

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=train_shuffle)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    print(f"trainer params: {cfg.trainer.params}")
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.trainer.resume_ckpt_path
    )


if __name__ == "__main__":
    main()
