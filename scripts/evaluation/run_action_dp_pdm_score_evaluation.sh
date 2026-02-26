TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/path/to/your/action_dp_agent.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navtest/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navtest/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=action_dp_agent \
    worker=ray_distributed \
    agent.checkpoint_path=$CHECKPOINT \
    experiment_name=action_dp_agent_eval \
    metric_cache_path=$CACHE_PATH \
    synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
    synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
