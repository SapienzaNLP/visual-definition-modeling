import argparse
from src.data_preparation.multimodal_datasets_lazy import FolderDataset

from torch.utils.data import DataLoader
from transformers import LxmertTokenizer

from src.data_preparation.multimodal_sampler import IndexSampler
from src.modeling.components import LxmertModelMod
from src.utils.modeling_frcnn import GeneralizedRCNN
from src.utils.frcnn_utils import Config, get_data
from src.utils.processing_image import Preprocess
import os
from tqdm import tqdm
import _pickle as pkl
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str,
                        default="/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/MultimodalGlosses/semeval2007.lxmert.npz")
    parser.add_argument("--batch_size", type=int,
                        default=4)

    args = parser.parse_args()
    out_file = args.out_file
    batch_size = args.batch_size

    lxmert_name = "unc-nlp/lxmert-base-uncased"
    encoder_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    lxmert = LxmertModelMod(lxmert_name).cuda()
    lxmert.eval()

    img_dataset = FolderDataset(
        hparams.img_dataset_path, name="img", shuffle=False)
    txt_dataset = FolderDataset(
        hparams.txt_dataset_path, name="txt", shuffle=False)
    txt_img_dataset = FolderDataset(
        hparams.txt_img_dataset_path, name="txt+img", shuffle=False)
    word_img_dataset = FolderDataset(
        hparams.word_img_dataset_path, name="word+img", shuffle=False)
    img_img_dataset = FolderDataset(
        hparams.img_img_dataset_path, name="img+img", shuffle=False)

    train_dataset = DatasetAlternator(
        lxmert_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        bart_tokenizer.pad_token_id,
        img_dataset,
        txt_dataset,
        txt_img_dataset,
        word_img_dataset,
        img_img_dataset
    )

    data_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=0,
                             collate_fn=train_dataset.get_batch_fun())

    ids = list()
    features = list()
    try:
        for batch in tqdm(data_loader, desc="Computing lxmert embeddings"):
            encoder_input = {'input_ids': batch['input_ids'].cuda(),
                        'attention_mask': batch['attention_mask'].cuda(),
                        'token_type_ids': batch['token_type_ids'].cuda(),
                        'visual_feats': batch['visual_feats'].cuda(), 'visual_pos': batch['visual_pos'].cuda(),
                        'visual_attention_mask': batch['visual_attention_mask'].cuda(), 
                        'output_hidden_states': True
                        }

            dict_out = lxmert(return_dict=True,
                        **encoder_input)

            visual_feats = dict_out['vision_hidden_states'][-1].detach().cpu().numpy()
            ids.extend(batch['ids'])
            features.extend(dict_out)
    except:
        print("Stopped")

    np.savez_compressed(out_file, ids=ids, features=features)

