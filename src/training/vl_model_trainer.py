from transformers import BertTokenizerFast
import wandb
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
    ModalitySampler, PicklesDataset,
)
from pytorch_lightning.callbacks import EarlyStopping
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
from src.modeling.vl_model_wrapper import get_lxmert_model, get_vl_bert_model, get_vil_bert_model, get_lxmert_pretrained_bart_model, get_vl_bert_pretrained_bart_model
from transformers import AutoTokenizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def prepare_encoder_input(batch):
    encoder_kwargs={
                        "input_imgs": batch["visual_feats"],
                        #"visual_attention_mask": batch["visual_attention_mask"],
                        "image_loc": batch["visual_pos"]
                    }
    return encoder_kwargs

class MGMFinetuner(pl.LightningModule):
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
        train_bart_only_steps=-1,
        train_img_only_steps=-1,
        joint_training=-1
    ):
        super().__init__()
        self.batch_kind_counter = Counter()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
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
        self.train_bart_only_steps = train_bart_only_steps
        self.train_img_only_steps = train_img_only_steps
        self.val_counter = Counter()

    def forward(self, batch):
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
            dev_loader = self.val_dataloader()
            steps = 0
            print()
            print("GENERATION STARTS " + "=" * 50)

            table = wandb.Table(
                columns=["Text", "Predicted Label", "True Label"])
            for batch in dev_loader:
                print(100 * "-" + "\nOutput:\n")
                batch = self.move_batch(
                    batch, self.trainer.precision, self.device)
                encoder_inputs = prepare_encoder_input(batch)
                beam_output = self.model.generate(
                    batch["input_ids"],
                    **self.generation_params,
                    decoder_start_token_id=self.decoder_tokenizer.cls_token_id,
                    decoder_kwargs={},
                    encoder_kwargs=encoder_inputs
                )
                
                data_entry = [self.tokenizer.decode(batch["input_ids"][0].tolist(), skip_special_tokens=True),
                              self.decoder_tokenizer.decode(
                                 beam_output[0].tolist(), skip_special_tokens=True),
                              self.decoder_tokenizer.decode(
                     batch["labels"][0].tolist() , skip_special_tokens=True).strip()]
                table.add_data(*data_entry)
                print(data_entry[0])
                print(data_entry[1])
                print(data_entry[2])
                steps += 1
                if steps == self.generate_samples:
                    if isinstance(self.dev_dataset, FolderDataset):
                        self.dev_dataset.stop()
                    break
            print()
            print("GENERATION ENDS " + "=" * 50)
            print()
            self.trainer.logger.experiment[0].log({"generations": table})
            self.val_counter = Counter()

    def training_step(self, batch, *args):
        train_bart = True
        train_img = True
        if self.global_step < self.train_bart_only_steps:
            train_img = False
        if self.train_img_only_steps > self.global_step >= self.train_bart_only_steps:
            train_bart = False 
        kind = list(set(batch["batch_kinds"]))[0]
        # try:   
        batch['train_bart'] = train_bart
        batch['train_img'] = train_img
        self.batch_kind_counter[kind] += 1
        self.log(
            kind, self.batch_kind_counter[kind], on_step=True, logger=False, prog_bar=True)
        outputs = self(batch)
        # except Exception as ex:
            # print(kind, batch["input_ids"].shape, batch["ids"])
            # print(ex)
            # sys.exit(1)
        loss = outputs[0]
        # self.log("train_loss", loss.item(), on_epoch=True)
        self.log("global_step", self.global_step, on_step=True)
        return loss

    def validation_step(self, batch, *args):
        outputs = self(batch)
        val_loss = outputs[0]
        self.log("val_loss", val_loss.item(), on_epoch=True, prog_bar=True)
        batch_kind = list(set(batch["batch_kinds"]))[0]
        self.val_counter[batch_kind] += batch['input_ids'].size(0)
        self.log(f"val_loss_{batch_kind}", val_loss.item(), on_epoch=True)
        return val_loss

    def test_step(self, batch, *args):
        outputs = self(batch)
        test_loss = outputs[0]
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        return test_loss

    def get_optimizer_and_scheduler(self):
        no_decay = self.hparams.no_decay_params

        decoder_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ]
                ,
                #+ [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
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
            initial_probabilities={"txt": 1.0,
                                   "txt+img": 0.0, "word+img": 0.0, "img+img": 0.0},
            target_probabilities={"txt": 0.25,
                                  "txt+img": 0.25, "word+img": 0.25, "img+img": 0.25},
            tasks_waiting_steps={"txt": 20000//g_acc,
                                 "txt+img": 20000//g_acc, "word+img": 40000//g_acc, "img+img": 60000//g_acc},
            tasks_target_steps={"txt": 100000//g_acc,
                                "txt+img": 100000//g_acc, "word+img": 100000//g_acc, "img+img": 100000//g_acc},
            train_bart_only_steps=self.hparams.train_bart_only_steps//g_acc,
            train_img_only_steps=self.hparams.train_img_only_steps//g_acc
        )

    def get_coherent_accumulation_sampler(self, dataset):
        return CircularSamplerWithCoherentAccumulation(
            dataset.num_datasets, self.hparams.gradient_accumulation
        )

    def train_dataloader(self):
        if self.training_scheme == "plain":
            sampler = self.get_plain_sampler(self.train_dataset)
        elif self.training_scheme == "coherent-accumulation":
            sampler = self.get_coherent_accumulation_sampler(
                self.train_dataset)
        elif self.training_scheme == "cv":
            sampler = self.get_cv_sampler(self.train_dataset)
        else:
            raise RuntimeError(
                "training scheme {} not found. Choose among {plain, cv, coherent-accumulation}.")

        modality_batch_sampler = ModalitySampler(sampler, self.hparams.batch_size)
        return DataLoader(
            self.train_dataset,
            batch_sampler=modality_batch_sampler,
            num_workers=0,
            collate_fn=self.train_dataset.get_batch_fun(),
        )

    def val_dataloader(self):
        if isinstance(self.dev_dataset, DatasetAlternator):
            self.dev_dataset.reset()
            sampler = self.get_plain_sampler(self.dev_dataset)
            modality_batch_sampler = ModalitySampler(sampler, self.hparams.batch_size)
            return DataLoader(
                self.dev_dataset,
                batch_sampler=modality_batch_sampler,
                num_workers=0,
                collate_fn=self.train_dataset.get_batch_fun(),
            )
        data_loader = DataLoader(
            self.dev_dataset,
            self.hparams.batch_size,
            num_workers=0,
            collate_fn=self.dev_dataset.get_batch_fun(),
        )

        return data_loader


def get_training_params(config_path):
    with open(config_path) as reader:
        training_params = json.load(reader)
    return TrainingParams(training_params)


if __name__ == "__main__":
    argparse = ArgumentParser()
    # argparse.add_argument("--exp_name", required=True, type=str)
    argparse.add_argument('--config', default='config/word+img_config.lxmert-pretrained-bart.json', type=str)
    argparse.add_argument("--offline", action="store_true")
    argparse.add_argument('--device', default='cuda', type=str)
    args = argparse.parse_args()
    # exp_name = args.exp_name
    offline = args.offline
    config_path = args.config
    arg_device = args.device
    hparams = get_training_params(config_path)
    if arg_device is not None:
        hparams.device = arg_device
    torch.manual_seed(hparams.random_seed)
    random.seed(hparams.random_seed)
    numpy.random.seed(hparams.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name)
    decoder_tokenizer = tokenizer
    if hparams.pretrained_decoder:
        decoder_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

    img_source = hparams.img_source

    task_datasets = list()
    print(f'Loading datsets for tasks: {hparams.tasks}')
    for task in hparams.tasks:
        if task == 'txt':
            aux = FolderDataset(
                hparams.txt_dataset_path, name="txt", img_source=hparams.img_source, shuffle=True, is_infinite=True)
        elif task == 'txt+img':
            aux = FolderDataset(
            hparams.txt_img_dataset_path, name="txt+img", img_source=hparams.img_source, shuffle=True, is_infinite=True)
        elif task == 'word+img':
            aux = FolderDataset(
            hparams.word_img_dataset_path, name="word+img", img_source=hparams.img_source, shuffle=True, is_infinite=True)
        elif task == 'img+img':
            aux = FolderDataset(
            hparams.img_img_dataset_path, name="img+img", img_source=hparams.img_source, shuffle=True, is_infinite=True)
        else:
            print(f'[ERROR] task {task} unrecognised, skipping.')
            continue
        task_datasets.append(aux)


    train_dataset = DatasetAlternator(
        tokenizer.pad_token_id,
        decoder_tokenizer.pad_token_id,
        img_source,
        *task_datasets

    )

    dev_dataset = PicklesDataset(
        hparams.dev_dataset_root, tokenizer.pad_token_id, hparams.batch_size, img_source)
    random.shuffle(dev_dataset.batches)
    print('loading', hparams.base_model)
    if not hparams.pretrained_decoder:
        if hparams.base_model == 'vl-bert':
            model = get_vl_bert_model() 
        elif hparams.base_model == 'vil-bert':
            model = get_vil_bert_model()
        elif hparams.base_model == 'lxmert':
            model = get_lxmert_model()
        else:
            raise RuntimeEerror(f'Cannot recognise base model {hparams.base_model}')
    else:
        if hparams.base_model == 'vl-bert':
            model = get_vl_bert_pretrained_bart_model() 
        elif hparams.base_model == 'lxmert':
            model = get_lxmert_pretrained_bart_model()
        else:
            raise RuntimeEerror(f'Cannot recognise base model {hparams.base_model}')

    model.to(hparams.device)
    wandb_logger = WandbLogger(
        hparams.model_name,
        project="multimodal_glosses",
        entity='research',
        offline=offline,
        log_model=True,
        save_dir=hparams.save_dir
    )
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    pprint(hparams)
    checkpointer = ModelCheckpoint(
        os.path.join(checkpoint_dir, "{global_step}"), monitor="val_loss",
        save_top_k=1
    )
    callbacks_store = []
    if hparams.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=hparams.monitor_var,
                mode=hparams.monitor_var_mode,
                patience=hparams.patience
            )
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
        callbacks=callbacks_store,
        log_every_n_steps=hparams.log_every_n_steps,
        val_check_interval=hparams.val_check_interval,
    )

    finetuner = MGMFinetuner(
        model,
        train_dataset,
        dev_dataset,
        hparams,
        generate_samples=20,
        train_encoder=False,
        infinite_iterators=hparams.infinite_iterators,
        decoder_tokenizer=decoder_tokenizer,
        training_scheme=hparams.training_scheme,
        train_bart_only_steps=hparams.train_bart_only_steps//hparams.gradient_accumulation,
        train_img_only_steps=hparams.train_img_only_steps//hparams.gradient_accumulation
    )
    trainer.fit(finetuner)
