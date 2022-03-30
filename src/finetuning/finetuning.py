import json
import os
from pprint import pprint
from src.data_preparation.multimodal_datasets_lazy import DatasetAlternator, FolderDataset
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.training.model_full_bart_trainer import MGMFinetuner
from src.modeling.model_full_bart_visual_feat import MGMFullBart
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
import torch
from transformers import AutoTokenizer
import re

def load_model(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.device))


    model = MGMFullBart.from_pretrained(
        config.bart_name, **config)

    state_dict = checkpoint["state_dict"]
    state_dict = {re.sub("model.", "", k, 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model = model.eval().to(config.device)
    return model


