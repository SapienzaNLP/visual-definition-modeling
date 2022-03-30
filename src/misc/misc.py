from torch.utils.data.dataloader import DataLoader
from src.modeling.components import TrainingParams
from src.modeling.model_full_bart_visual_feat import MGMFullBart
from transformers.tokenization_bart import BartTokenizerFast
from transformers.tokenization_lxmert import LxmertTokenizerFast
from src.data_preparation.multimodal_datasets_lazy import CircularSampler, DatasetAlternator, FolderDataset, ModalitySampler
import json

with open("config/pretraining_config.json") as reader:
        training_params = json.load(reader)
hparams =  TrainingParams(training_params)
lxmert_tokenizer = LxmertTokenizerFast.from_pretrained(hparams.lxmert_name)
bart_tokenizer = BartTokenizerFast.from_pretrained(hparams.bart_name)

img_img_dataset = FolderDataset(
        hparams.img_img_dataset_path, name="img+img", shuffle=True)

train_dataset = DatasetAlternator(
    lxmert_tokenizer.pad_token_id,
    bart_tokenizer.pad_token_id,
    bart_tokenizer.pad_token_id,
    img_img_dataset
)

sampler= CircularSampler(1)
modality_batch_sampler = ModalitySampler(sampler, 8)

dataloader = DataLoader(
            train_dataset,
            num_workers=0,
            batch_sampler=modality_batch_sampler,
            collate_fn=train_dataset.get_batch_fun(),
        )

mgm_bart = MGMFullBart.from_pretrained(
        hparams.bart_name, multimodal_encoder=hparams.lxmert_name, **hparams
    )

for batch in dataloader:
    mgm_bart(**batch)