TRAIN_TEST_SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=diffusion_transfuser_agent \
        agent.config.latent=True \
        experiment_name=training_diffusion_transfuser_agent  \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.params.max_epochs=100 \
        trainer.params.strategy=ddp_find_unused_parameters_true \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=False  \
        force_cache_computation=True 