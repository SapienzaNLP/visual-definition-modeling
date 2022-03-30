from src.baselines.lxmert_decoder import LxmertEncDec
from transformers import BartTokenizerFast
from transformers import LxmertTokenizerFast
from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast
import wandb
from src.modeling.model_full_bart_visual_feat import MGMFullBart
import os
import random
from argparse import ArgumentParser
from pprint import pprint
from typing import Counter
from src.data_preparation.multimodal_datasets_lazy import (
    CircularSampler,
    CircularSamplerWithCoherentAccumulation,
    CurriculumSampler,
    DatasetAlternator,
    FolderDataset,
    ModalitySampler,
    PicklesDataset,
)
import json

import numpy
import pytorch_lightning as pl
import torch
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from src.data_preparation.multimodal_datasets import (
    MultimodalTxtDataset,
)
from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator
from src.modeling.components import TrainingParams
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BaselineFinetuner(pl.LightningModule):
    def __init__(
        self,
        model,
        training_set,
        dev_set,
        training_params,
        test_dataloader=None,
        decoder_tokenizer=None,
        generate_samples=0,
        train_encoder=True,
        infinite_iterators=False,
        training_scheme=None,
    ):
        super().__init__()
        self.batch_kind_counter = Counter()
        self.model = model
        self.hparams = training_params
        self.train_dataset = training_set
        self.dev_dataset = dev_set
        self.test_dataloader = test_dataloader
        self.decoder_tokenizer = decoder_tokenizer
        self.generate_samples = generate_samples
        self.train_encoder = train_encoder
        self.infinite_iterators = infinite_iterators
        self.training_scheme = training_scheme
        self.generation_params = dict(
            num_beams=4,
            min_length=5,
            max_length=15,
            temperature=1,
            repetition_penalty=2,
            length_penalty=1.5,
            no_repeat_ngram_size=2,
        )

    def forward(self, batch):
        batch["input_ids"] = batch["crossmod_input_ids"]
        batch["attention_mask"] = batch["crossmod_attention_mask"]
        batch["token_type_ids"] = batch["crossmod_token_type_ids"]
        return self.model(**batch)

    def move_batch(self, batch, precision, device):
        for k, v in batch.items():
            if type(v) is not torch.Tensor:
                continue
            v = v.to(device)
            batch[k] = v
        return batch

    def on_validation_epoch_end(self) -> None:
        if self.generate_samples > 0:
            dev_loader = DataLoader(
                self.dev_dataset,
                batch_size=1,
                num_workers=1,
                collate_fn=self.dev_dataset.get_batch_fun(),
            )
            steps = 0
            print()
            print("GENERATION STARTS " + "=" * 50)

            table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
            for batch in dev_loader:
                batch = self.move_batch(batch, self.trainer.precision, self.device)

                beam_output = self.model.generate(
                    batch["crossmod_input_ids"],
                    **self.generation_params,
                    decoder_start_token_id=self.decoder_tokenizer.bos_token_id,
                    decoder_kwargs={},
                    encoder_kwargs={
                        "crossmod_input_ids": batch["crossmod_input_ids"],
                        "crossmod_attention_mask": batch["crossmod_attention_mask"],
                        "crossmod_token_type_ids": batch["crossmod_token_type_ids"],
                        "batch_kinds": batch["batch_kinds"],
                        "visual_feats": batch["visual_feats"],
                        "visual_pos": batch["visual_pos"],
                        "visual_attention_mask": batch["visual_attention_mask"],
                        "activate_img_features": batch["activate_img_features"],
                        "target_indexes": batch["target_indexes"],
                        "insert_img_objects": batch["insert_img_objects"],
                    },
                )
                print("Output:\n" + 100 * "-")
                data_entry = [
                    self.decoder_tokenizer.decode(batch["input_ids"][0].tolist()),
                    self.decoder_tokenizer.decode(beam_output[0].tolist()),
                    self.decoder_tokenizer.decode(batch["labels"][0].tolist()).strip(),
                ]
                table.add_data(*data_entry)
                print(data_entry[0])
                print(data_entry[1])
                print(data_entry[2])
                steps += 1
                if steps == self.generate_samples:
                    break
            print()
            print("GENERATION ENDS " + "=" * 50)
            print()
            trainer.logger.experiment[0].log({"generations": table})

    def training_step(self, batch, *args):
        kind = list(set(batch["batch_kinds"]))[0]
        try:
            self.batch_kind_counter[kind] += 1
            self.log(
                kind,
                self.batch_kind_counter[kind],
                on_step=True,
                logger=False,
                prog_bar=True,
            )
            outputs = self(batch)
        except Exception as ex:
            print(kind, batch["input_ids"].shape, batch["ids"])
            print(ex)
            sys.exit(1)
        loss = outputs[0]
        self.log("global_step", self.global_step, on_step=True)
        return loss

    def validation_step(self, batch, *args):
        outputs = self(batch)
        val_loss = outputs[0]
        self.log("val_loss", val_loss.item(), on_epoch=True, prog_bar=True)
        batch_kind = list(set(batch["batch_kinds"]))[0]
        self.log(f"val_loss_{batch_kind}", val_loss.item(), on_epoch=True)
        return val_loss

    def test_step(self, batch, *args):
        outputs = self(batch)
        test_loss = outputs[0]
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        return test_loss

    def get_optimizer_and_scheduler(self):
        no_decay = self.hparams.no_decay_params

        encoder_params = self.model.encoder.named_parameters()
        if not self.train_encoder:
            encoder_params = [
                (k, v) for k, v in encoder_params if not k.startswith("encoder.")
            ]

        decoder_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.decoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                + [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.decoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        decoder_optimizer = torch.optim.AdamW(
            decoder_optimizer_grouped_parameters, self.hparams.decoder_lr
        )

        return decoder_optimizer

    def configure_optimizers(self):
        return self.get_optimizer_and_scheduler()

    def get_plain_sampler(self, dataset):
        return CircularSampler(dataset.num_datasets)

    def get_cv_sampler(self, dataset):
        g_acc = self.hparams.gradient_accumulation
        return CurriculumSampler(
            dataset.num_datasets,
            self.hparams.batch_size,
            self.hparams.gradient_accumulation,
            convergence_function="linear",
            initial_probabilities={
                "img": 0.0,
                "txt": 1.0,
                "txt+img": 0.0,
                "word+img": 0.0,
                "img+img": 0.0,
            },
            target_probabilities={
                "img": 0.2,
                "txt": 0.2,
                "txt+img": 0.2,
                "word+img": 0.2,
                "img+img": 0.2,
            },
            tasks_waiting_steps={
                "img": 60000 // g_acc,
                "txt": 20000 // g_acc,
                "txt+img": 20000 // g_acc,
                "word+img": 40000 // g_acc,
                "img+img": 80000 // g_acc,
            },
            tasks_target_steps={
                "img": 100000 // g_acc,
                "txt": 100000 // g_acc,
                "txt+img": 100000 // g_acc,
                "word+img": 100000 // g_acc,
                "img+img": 100000 // g_acc,
            },
        )

    def get_coherent_accumulation_sampler(self, dataset):
        return CircularSamplerWithCoherentAccumulation(
            dataset.num_datasets, self.hparams.gradient_accumulation
        )

    def train_dataloader(self):
        if self.training_scheme == "plain":
            sampler = self.get_plain_sampler(self.train_dataset)
        if self.training_scheme == "coherent-accumulation":
            sampler = self.get_coherent_accumulation_sampler(self.train_dataset)
        if self.training_scheme == "cv":
            sampler = self.get_cv_sampler(self.train_dataset)
        else:
            raise RuntimeError(
                "training scheme {} not found. Choose among {plain, cv, coherent-accumulation}."
            )

        modality_batch_sampler = ModalitySampler(sampler, self.hparams.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_sampler=modality_batch_sampler,
            num_workers=0,
            collate_fn=self.train_dataset.get_batch_fun(),
        )

    def val_dataloader(self):
        samplers = [
            IndexSampler(self.dev_dataset.kind2indices[k])
            for k in self.dev_dataset.kind2indices.keys()
        ]

        alternator = SamplerAlternator(
            samplers,
            self.hparams.batch_size,
            drop_last=False,
            shuffle=False,
            infinite_iterators=False,
        )
        data_loader = DataLoader(
            self.dev_dataset,
            batch_sampler=alternator,
            num_workers=0,
            collate_fn=self.dev_dataset.get_batch_fun(),
        )

        return data_loader


def get_training_params():
    with open("config/baseline_config.json") as reader:
        training_params = json.load(reader)
    return TrainingParams(training_params)


if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--offline", action="store_true", default=False)
    args = argparse.parse_args()
    offline = args.offline
    hparams = get_training_params()
    torch.manual_seed(hparams.random_seed)
    random.seed(hparams.random_seed)
    numpy.random.seed(hparams.random_seed)

    encoder_tokenizer = LxmertTokenizerFast.from_pretrained(hparams.encoder_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(hparams.decoder_name, fast=True)

    img_dataset = FolderDataset(
        hparams.img_dataset_path, name="img", shuffle=True, is_infinite=True
    )
    txt_dataset = FolderDataset(
        hparams.txt_dataset_path, name="txt", shuffle=True, is_infinite=True
    )
    txt_img_dataset = FolderDataset(
        hparams.txt_img_dataset_path, name="txt+img", shuffle=True, is_infinite=True
    )
    word_img_dataset = FolderDataset(
        hparams.word_img_dataset_path, name="word+img", shuffle=True, is_infinite=True
    )
    img_img_dataset = FolderDataset(
        hparams.img_img_dataset_path, name="img+img", shuffle=True, is_infinite=True
    )

    train_dataset = DatasetAlternator(
        encoder_tokenizer.pad_token_id,
        decoder_tokenizer.pad_token_id,
        decoder_tokenizer.pad_token_id,
        img_dataset,
        txt_dataset,
        txt_img_dataset,
        word_img_dataset,
        img_img_dataset,
    )
    dev_dataset = PicklesDataset(
        hparams.dev_dataset_root, encoder_tokenizer.pad_token_id, hparams.batch_size
    )
    random.shuffle(dev_dataset.batches)

    baseline = LxmertEncDec()
    baseline.to(hparams.device)
    wandb_logger = WandbLogger(
        hparams.model_name,
        project="multimodal_glosses",
        offline=offline,
        log_model=True,
        save_dir=hparams.save_dir,
    )
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    pprint(hparams)
    checkpointer = ModelCheckpoint(
        os.path.join(checkpoint_dir, "{global_step}"), monitor="val_loss", save_top_k=3
    )
    if hparams.device == "cuda":
        gpus, precision = 1, 16
    else:
        gpus, precision = 0, 32
    trainer = pl.Trainer(
        gpus=gpus,
        precision=precision,
        max_steps=hparams.num_training_steps,
        checkpoint_callback=checkpointer,
        accumulate_grad_batches=hparams.gradient_accumulation,
        logger=[wandb_logger],
        gradient_clip_val=hparams.gradient_clip_val,
        num_sanity_val_steps=10,
        deterministic=True,
        log_every_n_steps=hparams.log_every_n_steps,
        val_check_interval=hparams.val_check_interval,
    )

    finetuner = BaselineFinetuner(
        baseline,
        train_dataset,
        dev_dataset,
        hparams,
        generate_samples=10,
        train_encoder=False,
        infinite_iterators=hparams.infinite_iterators,
        decoder_tokenizer=decoder_tokenizer,
        training_scheme=hparams.training_scheme,
    )
    trainer.fit(finetuner)