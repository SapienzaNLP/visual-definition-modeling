import json

from numpy.core.numeric import True_
from src.data_preparation.multimodal_datasets_lazy import DatasetAlternator, FiniteFolderDataset, FolderDataset
from src.finetuning.finetuning import load_model
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.utilities.parsing import AttributeDict
from src.training.model_full_bart_trainer import MGMFinetuner
import os
from pprint import pprint
from transformers import AutoTokenizer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

if __name__ == "__main__":
    config_path = "config/vqa_config.json"
    with open(config_path) as reader:
        config = json.load(reader)
    config = AttributeDict(config)
    bart_tokenizer = AutoTokenizer.from_pretrained(config.bart_name)
    lxmert_tokenizer = AutoTokenizer.from_pretrained(config.lxmert_name)
    model = load_model(config, config.checkpoint_to_load)
    
    img_dataset = FolderDataset(
        config.training_dataset_path, name="q&a", shuffle=False, is_infinite=True)
    
    training_set = DatasetAlternator(
        bart_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        img_dataset,
    )

    dev_dataset = FiniteFolderDataset(config.dev_dataset_path, name='q&a', shuffle=False,
            pad_token_id=bart_tokenizer.pad_token_id)
    
    finetuner = MGMFinetuner(
        model,
        training_set,
        dev_dataset,
        config,
        generate_samples=50,
        train_encoder=False,
        infinite_iterators=config.infinite_iterators,
        decoder_tokenizer=bart_tokenizer,
        training_scheme=config.training_scheme,
    )

    wandb_logger = WandbLogger(
        config.model_name,
        project="multimodal_glosses-vqa2",
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
