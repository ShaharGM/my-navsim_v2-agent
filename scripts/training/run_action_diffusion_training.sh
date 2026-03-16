NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

echo "----------------------------------------------------------------"
echo "GPUs Detected: $NUM_GPUS"
echo "----------------------------------------------------------------"

TRAIN_TEST_SPLIT=navmini

# Find a free port automatically
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Using master port: $MASTER_PORT"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=action_diffusion_agent \
        experiment_name=training_action_diffusion_agent \
        train_test_split=$TRAIN_TEST_SPLIT \
        dataloader.params.batch_size=8 \
        trainer.params.accumulate_grad_batches=1 \
        trainer.params.max_epochs=20000 \
        trainer.params.strategy=ddp_find_unused_parameters_true \
        agent.lr=1e-5 \
        +debug_overfit=True \
        agent.config.backbone_type="bev" \
        agent.config.freeze_backbone=True \
        agent.config.bev_ckpt=$NAVSIM_DEVKIT_ROOT/weights/gtrs_dp.ckpt \
        agent.config.vov_ckpt=$OPENSCENE_DATA_ROOT/models/dd3d_det_final.pth \
        agent.config.noise_type="ddpm" \
        agent.config.num_inference_proposals=100 \
        agent.config.num_diffusion_layers=5 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache_navmini/" \
        use_cache_without_dataset=True \
        force_cache_computation=False
