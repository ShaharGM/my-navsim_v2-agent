TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
	train_test_split=$TRAIN_TEST_SPLIT \
	agent=transfuser_forcer_agent \
	agent.config.latent=True \
	worker=ray_distributed \
	agent.checkpoint_path="/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/models/weights/transfuser_forcer_dual_latent_part_loss_seed_0.ckpt" \
	experiment_name=transfuser_forcer_agent_eval \
	synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
	synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
