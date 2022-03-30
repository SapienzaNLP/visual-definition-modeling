import os
from pprint import pprint
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers import LxmertTokenizer, BartTokenizer

from src.data_preparation.multimodal_datasets import MultimodalTxtDataset, ConditionedSampler, TxtImgBatchAlternator
from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator
from src.misc.loggers import IMGDropoutLogger
from src.modeling.components import TrainingParams
from src.modeling.model import Lxmert_LM
import logging

from src.modeling.model_bart_only import BartOnlyWrapper
from src.modeling.model_full_bart import MGM_full_bart


def get_constant_and_linear_decay_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int,
                                                       num_constant_steps: int, num_training_steps: int,
                                                       last_epoch: int = -1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        elif current_step < num_warmup_steps + num_constant_steps:
            return 1.0
        else:
            return max(
                0.0, float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_warmup_steps - num_constant_steps))
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class LxmertLMFinetuner(pl.LightningModule):

    def __init__(self, lxmertlm, training_set, dev_set, training_params, test_dataloader=None,
                 encoder_tokenizer=None, decoder_tokenizer=None, image_dropout_scheduler="linear",
                 generate_samples=0, train_encoder=True):
        super().__init__()
        self.lxmertlm = lxmertlm
        self.hparams = training_params
        self.train_dataset = training_set
        self.dev_dataset = dev_set
        self.test_dataloader = test_dataloader
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.img_dropout_scheduler = self.schedule_img_dropout if image_dropout_scheduler == "linear" else None
        self.generate_samples = generate_samples
        self.train_encoder = train_encoder
        self.generation_params = dict(num_beams=4,
                                      min_length=5,
                                      max_length=15,
                                      temperature=1,
                                      repetition_penalty=2,
                                      length_penalty=1.5,
                                      no_repeat_ngram_size=2)
        # self.steps_without_images = training_params.steps_without_images

    def forward(self, batch):
        return self.lxmertlm(**batch)

    def schedule_img_dropout(self):
        current_step = self.global_step
        if current_step < self.hparams.steps_without_images:
            self.lxmertlm.set_img_dropout(1.0)
        else:

            num_training_steps = self.hparams.num_training_steps
            num_warmup_steps = self.hparams.steps_without_images
            self.lxmertlm.set_img_dropout(max(
                self.lxmertlm.get_img_dropout_target(),
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
            )

    def adapt_batch(self, batch, precision, device):
        for k, v in batch.items():
            if type(v) is not torch.Tensor:
                continue
            v = v.to(device)
            # if precision == 16 and v.dtype == torch.float:
            #     v = v.half()
            batch[k] = v
        return batch

    def on_validation_epoch_end(self) -> None:
        if self.generate_samples > 0:
            dev_loader = DataLoader(self.dev_dataset, batch_size=1, num_workers=0,
                                    collate_fn=self.dev_dataset.get_batch_fun())
            steps = 0
            print()
            print("GENERATION STARTS " + "=" * 50)
            for batch in dev_loader:
                batch = self.adapt_batch(batch, self.trainer.precision, self.device)

                beam_output = self.lxmertlm.generate(
                    batch["input_ids"].cuda(),
                    **self.generation_params,
                    decoder_start_token_id=decoder_tokenizer.bos_token_id,
                    decoder_kwargs={},
                    encoder_kwargs={"visual_feats": batch["visual_feats"].cuda(),
                                    "visual_pos": batch["visual_pos"].cuda(),
                                    "visual_attention_mask": batch["visual_attention_mask"].cuda()}

                )
                print("Output:\n" + 100 * '-')
                print(encoder_tokenizer.decode(batch["input_ids"][0]))
                print(decoder_tokenizer.decode(beam_output[0]))
                print(decoder_tokenizer.decode(batch["labels"][0]).strip())
                steps += 1
                if steps == self.generate_samples:
                    break
            print()
            print("GENERATION ENDS " + "=" * 50)
            print()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.img_dropout_scheduler is not None:
            self.img_dropout_scheduler()
        outputs = self(batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = outputs[0]
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        test_loss = outputs[0]
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        return test_loss

    def get_optimizer_and_scheduler(self):
        no_decay = self.hparams.no_decay_params

        decoder_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.lxmertlm.model.named_parameters() if
                           not any(nd in n for nd in no_decay)] + [self.lxmertlm.final_logits_bias],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.lxmertlm.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        decoder_optimizer = torch.optim.AdamW(decoder_optimizer_grouped_parameters, self.hparams.decoder_lr)

        # decoder_lr_scheduler = get_constant_and_linear_decay_schedule_with_warmup(
        #     # get_linear_schedule_with_warmup(  #
        #     optimizer=decoder_optimizer,
        #     num_warmup_steps=self.hparams.num_warmup_steps,
        #     num_constant_steps=self.hparams.num_constant_steps,
        #     num_training_steps=self.hparams.num_training_steps
        # )


        return decoder_optimizer
        #[{"interval": 'step', 'scheduler': decoder_lr_scheduler, 'name': 'decoder_adamw'}]

    def configure_optimizers(self):
        return self.get_optimizer_and_scheduler()

    def train_dataloader(self):
        txt_img_indices = self.train_dataset.txt_img_indices
        txt_only_indices = self.train_dataset.txt_only_indices
        txt_img_indices_sampler = IndexSampler(txt_img_indices)
        txt_only_indices_sampler = IndexSampler(txt_only_indices)

        alternator = SamplerAlternator([txt_img_indices_sampler, txt_only_indices_sampler], 4, False)
        data_loader = DataLoader(self.train_dataset, batch_sampler=alternator, num_workers=0,
                                 collate_fn=self.train_dataset.get_batch_fun())

        return data_loader

    def val_dataloader(self):
        txt_img_indices = self.dev_dataset.txt_img_indices
        txt_only_indices = self.dev_dataset.txt_only_indices
        txt_img_indices_sampler = IndexSampler(txt_img_indices)
        txt_only_indices_sampler = IndexSampler(txt_only_indices)

        alternator = SamplerAlternator([txt_img_indices_sampler, txt_only_indices_sampler], 4, False)
        data_loader = DataLoader(self.dev_dataset, batch_sampler=alternator, num_workers=0,
                                 collate_fn=self.dev_dataset.get_batch_fun())

        return data_loader


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


def get_training_params(dataset=None):
    batch_size = 4
    accumulation_steps = 8
    epochs = 20
    num_training_steps = (
            ((len(dataset) // (batch_size * accumulation_steps)) + 1) * epochs) if dataset is not None else 75000
    num_warmup_steps = num_training_steps * 0.2
    gradient_clip_val = 0.0
    steps_without_images = num_warmup_steps
    num_constant_steps = num_warmup_steps // 4
    img_dropout = 0.5

    training_params = TrainingParams(encoder_lr=3e-6,
                                     decoder_lr=3e-6,
                                     batch_size=batch_size,
                                     traing_dataset_len=len(dataset) if dataset is not None else None,
                                     # no_decay_params=['bias', 'LayerNorm.weight'],
                                     no_decay_params = [],
                                     weight_decay=0.00,
                                     num_warmup_steps=num_warmup_steps,
                                     num_constant_steps=num_constant_steps,
                                     num_training_steps=num_training_steps,
                                     gradient_accumulation=accumulation_steps,
                                     gradient_clip_val=gradient_clip_val,
                                     encoder_concat_img_features=True,
                                     steps_without_images=steps_without_images,
                                     img_dropout=img_dropout,
                                     lang_transformer_layers=2,
                                     lang_transformer_heads=4,
                                     img_transformer_layers=2,
                                     img_transformer_heads=4,
                                     joint_transformer_layers=4,
                                     joint_transformer_heads=8,
                                     encoder_feature_output=768
                                     )
    return training_params


# import yaml
if __name__ == "__main__":
    bart_name = "facebook/bart-base"

    tokenizer = BartTokenizer.from_pretrained(bart_name)
    encoder_tokenizer, decoder_tokenizer = tokenizer, tokenizer
    train_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                         decoder_tokenizer,
                                         "<define>", "</define>",
                                         "data/in/semcor.data.dm.all.txt",
                                         # "data/in/semcor.data.dm.subset.train.txt",
                                         "/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/MultimodalGlosses/babelpic.frcnn.semcor.all.npz",
                                         limit_sentences=100)
    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       decoder_tokenizer,
                                       "<define>", "</define>",
                                       # "data/in/semeval2007.data.dm.txt",
                                       "data/in/semcor.data.dm.all.txt",
                                       "/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/MultimodalGlosses/babelpic.frcnn.semcor.all.npz", limit_sentences=100)
    training_params = get_training_params(train_dataset)
    mgm_bart = BartOnlyWrapper.from_pretrained(bart_name)
    mgm_bart.cuda()
    wandb_logger = WandbLogger("mgm_full_bart_base_pre",
                               project="multimodal_glosses",
                               offline=True,
                               log_model=True,
                               save_dir="data/out/")
    lr_logger = LearningRateMonitor()
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    pprint(training_params)
    checkpointer = ModelCheckpoint(checkpoint_dir, monitor="val_loss", save_top_k=1)
    trainer = pl.Trainer(gpus=1, precision=16, max_steps=training_params.num_training_steps,
                         checkpoint_callback=checkpointer,
                         accumulate_grad_batches=training_params.gradient_accumulation,
                         logger=[wandb_logger], callbacks=[lr_logger],
                         gradient_clip_val=training_params.gradient_clip_val,
                         num_sanity_val_steps=0,
                         deterministic=True
                         )  # , limit_val_batches=0.1)

    finetuner = LxmertLMFinetuner(mgm_bart, train_dataset, dev_dataset, training_params,
                                  image_dropout_scheduler="linear", generate_samples=10,
                                  train_encoder=True,
                                  encoder_tokenizer=encoder_tokenizer)
    trainer.fit(finetuner)
