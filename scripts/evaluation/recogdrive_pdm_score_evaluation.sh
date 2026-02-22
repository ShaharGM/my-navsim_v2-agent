TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
	train_test_split=$TRAIN_TEST_SPLIT \
	agent=recogdrive_agent \
	worker=ray_distributed \
	worker.threads_per_node=4 \
	agent.checkpoint_path="/sci/labs/sagieb/dawndude/projects/recogdrive/pretrained_weights/Diffusion_Planner_For_8B.ckpt" \
	agent.vlm_path='/sci/labs/sagieb/dawndude/projects/recogdrive/pretrained_weights/ReCogDrive_VLM_8B' \
	agent.cam_type='single' \
	agent.grpo=False \
	agent.cache_hidden_state=True \
	agent.cache_mode=True \
	agent.vlm_type="internvl" \
	agent.dit_type="small" \
	agent.sampling_method="ddim" \
	experiment_name=recogdrive_agent_eval \
	metric_cache_path=$CACHE_PATH \
	synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
	synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
