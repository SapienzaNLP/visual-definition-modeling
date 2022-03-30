import argparse

import torch
from src.modeling.modeling_lxmert_mod import LxmertModelMod
import numpy as np
from numpy.lib.shape_base import _make_along_axis_idx
from transformers.tokenization_bart import BartTokenizerFast
from transformers.tokenization_lxmert import LxmertTokenizerFast
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import _pickle as pkl
import json
from torch.nn.utils.rnn import pad_sequence


def compute_lxmertembeddings(examples, batch_size_lxmert, lxmert, crossmod_encoder_tokenizer):
    num_examples = len(examples)
    for i in tqdm(range(0, num_examples, batch_size_lxmert), desc="Computing LxMERT embeddings"):
        batch_examples = examples[i: i+batch_size_lxmert]
        batched_crossmod_input_ids = pad_sequence([torch.Tensor(e['input_crossmod_ids']).long() for e in batch_examples], batch_first=True, 
        padding_value=crossmod_encoder_tokenizer.pad_token_id)
        visual_feats = torch.stack([torch.Tensor(e['visual_feats']) for e in batch_examples], 0)
        crossmod_token_type_ids = torch.zeros(batched_crossmod_input_ids.shape).long()
        crossmod_encoder_mask = (batched_crossmod_input_ids != crossmod_encoder_tokenizer.pad_token_id)
        encoder_input = {
            'input_ids': batched_crossmod_input_ids.cuda(),
            'visual_feats': visual_feats.cuda(),
            'visual_pos': torch.stack([torch.Tensor(e['visual_pos']) for e in batch_examples], 0).cuda(),
            'visual_attention_mask': torch.ones(len(visual_feats), len(visual_feats[0])).cuda(),
            'token_type_ids': crossmod_token_type_ids.cuda(),
            'attention_mask': crossmod_encoder_mask.cuda(),
        }

        dict_out = lxmert(return_dict=True, **encoder_input)

        visual_feats = dict_out['vision_output'].detach().cpu().numpy()
        for e, f in zip(batch_examples, visual_feats):
            e['lxmert_feats'] = f
    return examples


def prepare_batch(batch, crossmod_encoder_tokenizer):
    batch_vf = torch.stack([torch.Tensor(x['visual_feats']) for x in batch], 0)
    batch_vp = torch.stack([torch.Tensor(x['visual_pos']) for x in batch], 0)
    crossmod_ids = [torch.LongTensor(x['example']['input_crossmod_ids']) for x in batch]
    crossmod_ids = pad_sequence(crossmod_ids, batch_first=True)
    token_type_ids = torch.zeros_like(crossmod_ids)
    crossmod_mask = crossmod_ids != crossmod_encoder_tokenizer.pad_token_id
    return {
        'input_ids': crossmod_ids.cuda(),
        'visual_feats': batch_vf.cuda(),
        'visual_pos': batch_vp.cuda(),
        'visual_attention_mask': torch.ones(len(batch_vf), len(batch_vf[0])).cuda(),
        'token_type_ids': token_type_ids.cuda(),
        'attention_mask': crossmod_mask.cuda(),
    }

def merge_new_features(output_dict, batch):
    visual_feats = output_dict['vision_output'].tolist()
    for x, vf in zip(batch, visual_feats):
        l = x['example']['lxmert_feats'] if x['example']['lxmert_feats'] is not None else  list()
        l.append(np.array(vf))
        x['example']['lxmert_feats'] = l

def compute_lxmertembeddings_multi_images(examples, batch_size_lxmert, lxmert, crossmod_encoder_tokenizer):
    """
    Modifies examples in-place such data x['visual_feats'] (where x is an element of examples) contains lxmert-extracted
    features.
    """
    batch = list()
    
    for i, example in tqdm(enumerate(examples), desc='computing lxmert embeddings'):
        all_visual_feats = list(example['visual_feats'])∏
        all_visual_pos = list(example['visual_pos'])
        for vf, vp in zip(all_visual_feats, all_visual_pos):
            batch.append({'example_idx':i, 'visual_feats':vf, 'visual_pos':vp, 'example': example})
            if len(batch) == batch_size_lxmert:
                ready_batch = prepare_batch(batch, crossmod_encoder_tokenizer)
                dict_output = lxmert(return_dict=True, **ready_batch)
                merge_new_features(dict_output, batch)
                batch = list()
    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, nargs='*',
                        default=[
                        #'/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.train.frcnn.all_imgs.0.npz',
                        #'/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.train.frcnn.all_imgs.1.npz',
                        #'/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.train.frcnn.all_imgs.2.npz',
                        #'/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.train.frcnn.all_imgs.3.npz',
                        '/media/bianca/fluffy_potato/MultimodalGlosses/babelpic.frcnn.all_imgs.other.0.npz',
                        '/media/bianca/fluffy_potato/MultimodalGlosses/babelpic.frcnn.all_imgs.other.1.npz',
                        '/media/bianca/fluffy_potato/MultimodalGlosses/babelpic.frcnn.all_imgs.other.2.npz',
                        '/media/bianca/fluffy_potato/MultimodalGlosses/babelpic.frcnn.all_imgs.other.3.npz'])
    parser.add_argument("--dm_file", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/semcor.data.dm.new_gloss.txt')
    parser.add_argument("--babelpic_bns_file", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/babelpic_bns.pkl')
    parser.add_argument("--out_dir", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/training_multi_image/')
    parser.add_argument("--start_def_token", type=str,
                        default='<define>')
    parser.add_argument("--end_def_token", type=str,
                        default='</define>')
    parser.add_argument("--batch_size", type=int,
                        default=100)
    parser.add_argument("--cache_lxmert", type=bool,
                        default=True)
    parser.add_argument("--batch_size_lxmert", type=int,
                        default=4)

    args = parser.parse_args()
    features_file = args.features_file
    dm_file = args.dm_file
    babelpic_bns_file = args.babelpic_bns_file
    out_dir = args.out_dir
    start_def_token = args.start_def_token
    end_def_token = args.end_def_token
    batch_size = args.batch_size
    cache_lxmert = args.cache_lxmert
    batch_size_lxmert = args.batch_size_lxmert

    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    crossmod_encoder_tokenizer = LxmertTokenizerFast.from_pretrained(lxmert_name)
    decoder_tokenizer = AutoTokenizer.from_pretrained(bart_name, fast=True)
    lxmert = LxmertModelMod.from_pretrained(lxmert_name).cuda()
    lxmert.eval()

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    babelpic_bns = set()
    with open(babelpic_bns_file, 'rb') as reader:
        babelpic_bns = pkl.load(reader)

    kind2examples = dict()
    img2indices = dict()
    with open(dm_file) as lines:
        for i, line in enumerate(tqdm(lines, desc="Reading input file")):
            fields = line.split("\t")
            (
                id,
                sentence,
                target_word,
                token_indices,
                target_bnid,
                gloss,
                bnids,
            ) = fields
            start_token, end_token = [int(x) for x in token_indices.split("-")]
            bnids = bnids.split(",")
            gloss = gloss.strip().capitalize()
            if not gloss.endswith("."):
                gloss = gloss + "."
            gloss_ids = decoder_tokenizer.encode(gloss)
            kind = "txt"
            if target_bnid in babelpic_bns:
                kind = "txt+img"
            if id.startswith("word+img"):
                kind = "word+img"
            elif id.startswith("caption"):
                kind = "img"
            elif id.startswith("img+img"):
                kind = "img+img"
            # elif id.startswith("q&a"):
                # kind = "q&a"
            if kind == "img" or kind == "img+img":
                sentence = ""
            # elif kind == "q&a":
                # sentence = sentence.strip()
            else:
                sentence = sentence.strip().split(" ")
                sentence = " ".join(
                    sentence[:start_token]
                    + [start_def_token]
                    + sentence[start_token:end_token]
                    + [end_def_token]
                    + sentence[end_token:]
                )
            sentence_img_ids = crossmod_encoder_tokenizer.encode(sentence)
            if kind == "img+img":
                sentence = start_def_token + " " + end_def_token
            sentence = kind + ": " + sentence
            sentence_ids = decoder_tokenizer.encode(sentence)

            examples = kind2examples.get(kind, list())
            examples.append(
                dict(
                    id=id,
                    input_ids=sentence_ids,
                    input_crossmod_ids=sentence_img_ids,
                    target_bnid=target_bnid,
                    decoder_input_ids=gloss_ids,
                    bnids=bnids,
                    kind=kind,
                    visual_feats=None,
                    visual_pos=None,
                    index=start_token,
                    lxmert_feats=None
                )
            )
            kind2examples[kind] = examples
            indices = img2indices.get(target_bnid, list())
            indices.append((kind, len(examples)-1))
            img2indices[target_bnid] = indices

    for k in kind2examples:
        kind_dir = os.path.join(out_dir, k)
        if not os.path.isdir(kind_dir):
            os.makedirs(kind_dir)
        with open(os.path.join(out_dir, k, 'metadata.json'), 'w') as writer:
            json.dump({'examples': len(kind2examples[k])}, writer)

    txt_examples = kind2examples['txt'] if 'txt' in kind2examples else None
    kind2examples_images = dict()
    kind2indices = dict()
    for f in features_file:
        print(f)
        data = np.load(f)
        for s, boxes, feats in tqdm(zip(data.get('synsets'), data.get('normalized_boxes'), data.get('features')), desc="Reading features file"):
            img_id = s.split('_')[0]
            if img_id not in img2indices:
                continue
            for kind, idx in img2indices[img_id]:
                ex = kind2examples[kind][idx]
                visual_feats = ex['visual_feats'] if ex['visual_feats'] is not None else list()
                visual_pos  = ex['visual_pos'] if ex['visual_pos'] is not None else list()
                visual_feats.append(feats)
                visual_pos.append(boxes)
                ex['visual_feats'] = visual_feats
                ex['visual_pos'] = visual_pos
                examples = kind2examples_images.get(kind, list())
                examples.append(ex)
                if (len(examples) % batch_size) == 0:
                    id_file = kind2indices.get(kind, 0)
                    if cache_lxmert:
                        # examples = compute_lxmertembeddings(examples, batch_size_lxmert, lxmert, crossmod_encoder_tokenizer)
                        examples = compute_lxmertembeddings_multi_images(examples, batch_size_lxmert, lxmert, crossmod_encoder_tokenizer)
                    with open(os.path.join(out_dir, kind, 'pkl_{}.pkl'.format(id_file)), 'wb') as writer:
                        pkl.dump(examples, writer)  
                    id_file += 1
                    kind2indices[kind] = id_file    
                    del examples
                    examples = list()
                kind2examples_images[kind] = examples  
        del data
        # img2indices.pop(img_id)
    
    del kind2examples

    for kind in kind2examples_images:
        if len(kind2examples_images[kind]):
            id_file = kind2indices.get(kind, 0)
            examples = kind2examples_images[kind]
            if cache_lxmert:
                examples = compute_lxmertembeddings_multi_images(examples, batch_size_lxmert, lxmert, crossmod_encoder_tokenizer)
            with open(os.path.join(out_dir, kind, 'pkl_{}.pkl'.format(id_file)), 'wb') as writer:
                        pkl.dump(examples, writer)
    
    if txt_examples and len(txt_examples) > 0:
        with open(os.path.join(out_dir, 'txt/pkl_0.pkl'), 'wb') as writer:
            pkl.dump(txt_examples, writer)
