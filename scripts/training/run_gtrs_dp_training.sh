NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "----------------------------------------------------------------"
echo "GPUs Detected: $NUM_GPUS"
echo "----------------------------------------------------------------"

TRAIN_TEST_SPLIT=navmini

torchrun --nproc_per_node=$NUM_GPUS \
        $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=gtrs_diffusion_policy \
        experiment_name=training_gtrs_dp_agent  \
        train_test_split=$TRAIN_TEST_SPLIT \
        dataloader.params.batch_size=8 \
        trainer.params.accumulate_grad_batches=1 \
        trainer.params.max_epochs=1000 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 