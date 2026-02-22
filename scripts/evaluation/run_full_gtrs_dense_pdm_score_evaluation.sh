TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
	train_test_split=$TRAIN_TEST_SPLIT \
	agent=full_gtrs_dense_agent \
	worker=ray_distributed \
	agent.dp_checkpoint_path="/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/models/weights/gtrs_dp_seed_0_navtrain_retrained.ckpt" \
	agent.scorer_checkpoint_path="/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/models/gtrs_dense_vov.ckpt" \
	agent.hydra_config.vocab_path=${NAVSIM_DEVKIT_ROOT}/traj_final/8192.npy \
	experiment_name=full_gtrs_dense_agent_eval \
	synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
	synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
