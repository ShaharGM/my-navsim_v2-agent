TRAIN_TEST_SPLIT=navmini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=transfuser_forcer_agent \
        agent.config.latent=True \
        experiment_name=training_transfuser_forcer_agent  \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=False  \
        force_cache_computation=True 