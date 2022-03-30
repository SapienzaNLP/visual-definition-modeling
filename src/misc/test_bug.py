from transformers import BartTokenizerFast
from transformers import LxmertTokenizerFast
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
    ModalitySampler, PicklesDataset,
)
from tqdm import tqdm
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
from src.modeling.components import TrainingParams, prepare_encoder_input
from torch.utils.data.dataloader import DataLoader
def get_training_params():
    with open("config/pretraining_config.json") as reader:
        training_params = json.load(reader)
    return TrainingParams(training_params)

def get_cv_sampler(dataset, hparams):
    g_acc = hparams.gradient_accumulation
    sampler= CurriculumSampler(
        dataset.num_datasets,
        hparams.batch_size,
        hparams.gradient_accumulation,
        convergence_function="linear",
        initial_probabilities={"txt": 1.0,
                                "txt+img": 0.0, "word+img": 0.0, "img+img": 0.0},
        target_probabilities={"txt": 0.25,
                                "txt+img": 0.25, "word+img": 0.25, "img+img": 0.25},
        tasks_waiting_steps={"txt": 20000//g_acc,
                                "txt+img": 20000//g_acc, "word+img": 40000//g_acc, "img+img": 60000//g_acc},
        tasks_target_steps={"txt": 100000//g_acc,
                            "txt+img": 100000//g_acc, "word+img": 100000//g_acc, "img+img": 100000//g_acc},
        train_bart_only_steps=hparams.train_bart_only_steps//g_acc,
        train_img_only_steps=hparams.train_img_only_steps//g_acc
    )
    modality_batch_sampler = ModalitySampler(sampler, hparams.batch_size)
    return DataLoader(
        dataset,
        batch_sampler=modality_batch_sampler,
        num_workers=0,
        collate_fn=dataset.get_batch_fun(),
    )

if __name__ == '__main__':
    hparams = get_training_params()
    txt_dataset = FolderDataset(
        hparams.txt_dataset_path, name="txt", shuffle=True, is_infinite=True)
    txt_img_dataset = FolderDataset(
        hparams.txt_img_dataset_path, name="txt+img", shuffle=True, is_infinite=True)
    word_img_dataset = FolderDataset(
        hparams.word_img_dataset_path, name="word+img", shuffle=True, is_infinite=True)
    img_img_dataset = FolderDataset(
        hparams.img_img_dataset_path, name="img+img", shuffle=True, is_infinite=True)
    bart_tokenizer = BartTokenizerFast.from_pretrained(hparams.bart_name)

    train_dataset = DatasetAlternator(
        bart_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        #img_dataset,
        txt_dataset,
        txt_img_dataset,
        word_img_dataset,
        img_img_dataset
    )

    dataloader = get_cv_sampler(train_dataset, hparams)
    for batch in tqdm(dataloader):
        pass