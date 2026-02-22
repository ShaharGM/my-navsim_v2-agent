TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=weighted_ego_recogdrive_agent \
agent.recogdrive_checkpoint_path="/sci/labs/sagieb/dawndude/projects/recogdrive/pretrained_weights/Diffusion_Planner_For_8B.ckpt" \
agent.vlm_path="/sci/labs/sagieb/dawndude/projects/recogdrive/pretrained_weights/ReCogDrive_VLM_8B" \
experiment_name=training_we_recogdrive_agent \
train_test_split=$TRAIN_TEST_SPLIT \
