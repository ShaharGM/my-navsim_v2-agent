NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "----------------------------------------------------------------"
echo "GPUs Detected: $NUM_GPUS"
echo "----------------------------------------------------------------"

TRAIN_TEST_SPLIT=navtrain

# Find a free port automatically
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using master port: $MASTER_PORT"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=action_dp_agent \
        experiment_name=training_action_dp_agent  \
        train_test_split=$TRAIN_TEST_SPLIT \
        dataloader.params.batch_size=32 \
        trainer.params.accumulate_grad_batches=2 \
        trainer.params.max_epochs=75 \
        trainer.params.strategy=ddp_find_unused_parameters_true \
        agent.lr=1e-5 \
        +agent.config.scheduler='default' \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False \
        trainer.resume_ckpt_path="${NAVSIM_EXP_ROOT}/training_action_dp_agent/2026.02.22.17.04.07/lightning_logs/version_29429554/checkpoints/epoch_048-val_loss_1.9186.ckpt"