TRAIN_TEST_SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=weighted_ego_status_mlp_agent \
agent.lr=0.001 \
agent.mlp_checkpoint_path="/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/models/weights/ego_status_mlp_seed_0.ckpt" \
experiment_name=training_we_mlp_agent \
trainer.params.max_epochs=50 \
train_test_split=$TRAIN_TEST_SPLIT \
+trainer.params.devices=1 \
