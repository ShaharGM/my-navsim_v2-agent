TRAIN_TEST_SPLIT=navhard_two_stage
CHECKPOINT=/sci/labs/sagieb/dawndude/projects/navsim_workspace/exp/training_action_dp_agent/2026.02.22.17.04.07/lightning_logs/version_29429554/checkpoints/epoch_048-val_loss_1.9186.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=action_dp_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=action_dp_agent_eval \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
