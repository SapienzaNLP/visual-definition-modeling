if __name__ == "__main__":
    # argparse = ArgumentParser()
    # argparse.add_argument("--exp_name", required=True)
    # args = argparse.parse_args()
    config_path = "config/vqa_config.json"
    with open(config_path) as reader:
        config = json.load(reader)
    config = AttributeDict(config)
    bart_tokenizer = AutoTokenizer.from_pretrained(config.bart_name)
    lxmert_tokenizer = AutoTokenizer.from_pretrained(config.lxmert_name)
    model = load_model(config, config.checkpoint_to_load)

    
    img_dataset = FolderDataset(
        config.img_dataset_path, name="vqa", shuffle=True)
    
    training_set = DatasetAlternator(
        lxmert_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        img_dataset,
    )


    dev_set = MultimodalTxtDataset(
        bart_tokenizer,
        bart_tokenizer,
        lxmert_tokenizer,
        config.bart_name,
        config.bart_name,
        config.lxmert_name,
        config.start_define,
        config.end_define,
        config.dev_txt_path,
        config.dev_img_path,
        limit_sentences=-1,
        is_infinite=False,
    )
    finetuner = MGMFinetuner(
        model,
        training_set,
        dev_set,
        config,
        generate_samples=10,
        train_encoder=False,
        infinite_iterators=config.infinite_iterators,
        decoder_tokenizer=bart_tokenizer,
        training_scheme=config.training_scheme,
    )

    wandb_logger = WandbLogger(
        config.model_name,
        project="multimodal_glosses-definition-modelling-vqa",
        offline=False,
        log_model=True,
        save_dir=config.save_dir,
    )
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    pprint(config)
    checkpointer = ModelCheckpoint(
        os.path.join(checkpoint_dir, "{global_step}"), monitor="val_loss", save_top_k=1
    )

    if config.device == "cuda":
        gpus, precision = 1, 16
    else:
        gpus, precision = 0, 32

    trainer = pl.Trainer(
        gpus=gpus,
        precision=precision,
        max_steps=config.num_training_steps,
        checkpoint_callback=checkpointer,
        accumulate_grad_batches=config.gradient_accumulation,
        logger=[wandb_logger],
        gradient_clip_val=config.gradient_clip_val,
        num_sanity_val_steps=10,
        deterministic=True,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
    )
    trainer.fit(finetuner)