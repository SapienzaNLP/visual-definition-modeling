import logging
import random
from typing import Dict, List
import pickle as pkl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BartTokenizer
import hashlib

random.seed(34)
import numpy as np
import os
CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

np.random.seed(23)
pad_image_features = np.zeros((36, 2048))
pad_image_boxes = np.zeros((36, 4))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_img_features_synsets_index(synsets_img_idx):
    index = dict()
    for i, s in enumerate(synsets_img_idx):
        s = s.split("_")[0]
        if s not in index:
            index[s] = set()
        index[s].add(i)
    return index

import hashlib 
import pickle as pkl
class MultimodalTxtDataset(Dataset):
    def __init__(self, encoder_tokenizer: PreTrainedTokenizer,
                 decoder_tokenizer: PreTrainedTokenizer, 
                 crossmod_encoder_tokenizer: PreTrainedTokenizer,
                 encoder_name, decoder_name, crossmod_name,
                 start_def_token, end_def_token, txt_path: str,
                 img_features_path: str,
                 limit_sentences: int = -1,
                 is_infinite=False):
        self.examples = list()
        self.is_infinite=is_infinite
        self.encoder_pad_token_id = encoder_tokenizer.pad_token_id
        self.decoder_pad_token_id = decoder_tokenizer.pad_token_id
        self.crossmod_encoder_pad_token_id = crossmod_encoder_tokenizer.pad_token_id
        self.kind2indices = {"txt+img":list(), "txt":list(), "word+img":list(), "img": list(), "img+img": list()}

        # self.txt_only_indices = list()
        # self.txt_img_indices = list()
        cache_file_name = txt_path + img_features_path + encoder_name + decoder_name + crossmod_name
        cache_file_name = hashlib.md5(bytes(cache_file_name, "utf8")).hexdigest()
        if os.path.exists(os.path.join(CACHE_DIR, cache_file_name)):
            logger.info("found cached dataset: {}".format(cache_file_name))
            with open(os.path.join(CACHE_DIR, cache_file_name), "rb") as reader:
                self.examples = pkl.load(reader)
                if limit_sentences > 0:
                    self.examples = self.examples[:limit_sentences]
                self.indicize_examples()
                return
        
        img_features_files = np.load(img_features_path)
        synsets_img_idx = img_features_files["synsets"]
        img_features_index = build_img_features_synsets_index(synsets_img_idx)
        all_img_features = img_features_files["features"]
        all_img_boxes = img_features_files["normalized_boxes"]
        target_synset_without_bnid = 0
        with open(txt_path) as lines:
            for i, line in enumerate(tqdm(lines)):
                fields = line.split("\t")
                id, sentence, target_word, token_indices, target_bnid, gloss, bnids = fields
                start_token, end_token = [int(x) for x in token_indices.split("-")]
                bnids = bnids.split(",")
                gloss = gloss.strip().capitalize()
                if not gloss.endswith("."):
                    gloss = gloss + "."
                gloss_ids = decoder_tokenizer.encode(gloss)
                img_boxes, img_features = self.get_image_features(target_bnid, img_features_index, all_img_features,
                                                                  all_img_boxes)
                # prefix = "<sentence&image>"
                kind = 'txt'
                if img_features is None:
                    target_synset_without_bnid += 1
                    img_boxes, img_features = pad_image_boxes, pad_image_features
                    visual_attention_mask = np.zeros(*img_features.shape[:-1])
                else:
                    visual_attention_mask = np.ones(*img_features.shape[:-1])
                    kind = 'txt+img'
                if id.startswith("word+img"):
                     kind = "word+img"
                elif id.startswith("caption"):
                    kind = "img"
                elif id.startswith("img+img"):
                    kind = "img+img"

                if kind == "img" or kind == "img+img":
                    sentence = ""
                else:
                    sentence = sentence.strip().split(" ")
                    sentence = " ".join(sentence[:start_token] + [start_def_token] + sentence[start_token:end_token] + \
                                        [end_def_token] + sentence[end_token:])
                sentence_img_ids = crossmod_encoder_tokenizer.encode(sentence)
                if kind == "img+img":
                    sentence = start_def_token + " " + end_def_token
                sentence = kind + ": " + sentence
                sentence_ids = encoder_tokenizer.encode(sentence)

                self.examples.append(dict(id=id, input_ids=sentence_ids, input_crossmod_ids=sentence_img_ids,
                                          target_bnid=target_bnid,
                                          decoder_input_ids=gloss_ids, bnids=bnids, img_boxes=img_boxes,
                                          img_features=img_features, visual_attention_mask=visual_attention_mask,
                                          kind=kind, index=start_token,))
                if limit_sentences > 0 and len(self.examples) == limit_sentences:
                    break
        
        print("target synsets without images {}".format(target_synset_without_bnid))
        self.examples = sorted(self.examples, key=lambda x: -len(x["input_ids"]))
        self.indicize_examples()
        with open(os.path.join(CACHE_DIR, cache_file_name), "wb") as writer:
            logger.info("dumping dataset to: {}".format(cache_file_name))
            pkl.dump(self.examples, writer)

    def indicize_examples(self):
        for i, ex in enumerate(self.examples):
            k = ex["kind"]
            self.kind2indices[k].append(i)
            # if k == "txt":
                # self.txt_only_indices.append(i)
            # else:
                # self.txt_img_indices.append(i)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        if self.is_infinite:
            raise NotImplementedError()
        return len(self.examples)

    def get_image_features(self, synset_id, img_features_index, img_features, img_boxes):
        # return normalized_boxes, features
        if synset_id not in img_features_index:
            return None, None
        indices = list(img_features_index[synset_id])
        idx = random.randint(0, len(indices) - 1)
        img_index = indices[idx]
        features = img_features[img_index]
        boxes = img_boxes[img_index]
        return boxes, features

    def get_batch_fun(self):
        def collate_fn(examples):
            ids, input_ids, input_crossmod_ids, img_features, img_pos, visual_attention_mask, gloss_ids, kinds, indexes = zip(
                *[(e["id"],
                   torch.Tensor(e["input_ids"]).long(), torch.Tensor(e["input_crossmod_ids"]).long(),
                   e["img_features"], e["img_boxes"],
                   e["visual_attention_mask"],
                   torch.Tensor(e["decoder_input_ids"]).long(), e["kind"], e["index"])
                  for e in examples])
            batched_img_features = torch.Tensor(img_features)
            batched_visual_attention_mask = torch.Tensor(visual_attention_mask)
            batched_img_pos = torch.Tensor(img_pos)
            batched_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.encoder_pad_token_id)
            encoder_mask = batched_input_ids != self.encoder_pad_token_id
            token_type_ids = torch.ones(batched_input_ids.shape).long()

            # TODO crossmod_token_type_ids should be 0 !!!!!!!!!
            batched_crossmod_input_ids = pad_sequence(input_crossmod_ids, batch_first=True, padding_value=self.crossmod_encoder_pad_token_id)
            crossmod_encoder_mask = batched_crossmod_input_ids != self.crossmod_encoder_pad_token_id
            crossmod_token_type_ids = torch.zeros(batched_crossmod_input_ids.shape).long()

            batched_gloss_ids = pad_sequence(gloss_ids, batch_first=True, padding_value=self.decoder_pad_token_id)
            decoder_input_ids = batched_gloss_ids[:, :-1]
            labels = batched_gloss_ids[:, 1:]
            decoder_mask = decoder_input_ids == self.decoder_pad_token_id
            if kinds is not None:
                assert len(set(kinds)) == 1 or logging.error(
                    "cannot handle batches with example of mixed kinds (txt and txt+img)"
                )
                activate_img_features = list(set(kinds))[0] == "txt+img"
            else:
                activate_img_features = True
            # assert len(set(kinds)) == 1 or print("error")
            if kinds is not None:
                assert len(set(kinds)) == 1 or logger.error(
                    "cannot handle batches with example of mixed kinds (txt and txt+img)")
                #activate_img_features = list(set(kinds))[0] == "txt+img"
                activate_img_features = "img" in list(set(kinds))[0]
            else:
                activate_img_features = True

            insert_img_object = "img+img" in list(set(kinds))[0]

            return {"ids": ids,
                    "input_ids": batched_input_ids,
                    "attention_mask": encoder_mask,
                    "token_type_ids": token_type_ids,
                    "crossmod_input_ids": batched_crossmod_input_ids,
                    "crossmod_attention_mask": crossmod_encoder_mask,
                    "crossmod_token_type_ids": crossmod_token_type_ids,
                    "visual_feats": batched_img_features,
                    "visual_pos": batched_img_pos,
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_padding_mask": decoder_mask,
                    "visual_attention_mask": batched_visual_attention_mask,
                    "labels": labels.contiguous(),
                    "batch_kinds": kinds,
                    "activate_img_features": activate_img_features,
                    "insert_img_objects": insert_img_object,
                    "target_indexes": indexes,}

        return collate_fn


class ConditionedSampler(Sampler):
    def __init__(self, indices: Dict[str, List[int]], current_index_name: str,
                 infinite: bool = True):
        self.indices = indices
        """
        txt: [1 2 4 6 8]
        img: [3 5 7 9 10]
        """
        self.step = {k: 0 for k in self.indices.keys()}
        self.infinite = infinite
        self.current_index_name = current_index_name
        self.len = sum([len(self.indices[k]) for k in self.indices.keys()])

    def __len__(self):
        return self.len

    def __iter__(self):
        while True:
            current_index_name = self.current_index_name
            indices = self.indices[current_index_name]
            idx = self.step[current_index_name]
            yield indices[idx]
            self.step[current_index_name] += 1
            if idx + 1 >= len(indices):
                random.shuffle(indices)

                if self.infinite:
                    self.step[current_index_name] = 0
                else:
                    if all(self.step[k] >= len(self.indices[k]) for k in self.indices):
                        raise StopIteration()
                    else:
                        yield -1
