TRAIN_TEST_SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=diffusion_forcing_agent \
experiment_name=training_diffusion_forcing_agent \
train_test_split=$TRAIN_TEST_SPLIT \
trainer.params.max_epochs=50 \
cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
use_cache_without_dataset=True  \
force_cache_computation=False 