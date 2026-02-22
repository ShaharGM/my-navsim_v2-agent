TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
	train_test_split=$TRAIN_TEST_SPLIT \
	agent=diffusiondrive_agent \
	worker=ray_distributed \
	agent.checkpoint_path="/sci/labs/sagieb/dawndude/projects/DiffusionDrive/models/weights/diffusiondrive_navsim_88p1_PDMS" \
	experiment_name=diffusiondrive_agent_eval_one_stage \
	synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
	synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
