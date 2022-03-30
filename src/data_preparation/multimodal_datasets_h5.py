import hashlib
import logging
import pickle as pkl
import random
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import h5py

random.seed(34)
import os

import numpy as np

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


def build_img_features_synsets_index(synsets_img_idx):
    index = dict()
    for i, s in enumerate(synsets_img_idx):
        s = str(s, "utf8")
        s = s.split("_")[0]
        if s not in index:
            index[s] = set()
        index[s].add(i)
    return index

def get_image_indices(synset_id, img_features_index):
    if synset_id not in img_features_index:
            return None
    return list(img_features_index[synset_id])

def get_image_features(
        synset_id, img_features_index, img_features, img_boxes, num_of_images=5
    ):
        # return normalized_boxes, features
        if synset_id not in img_features_index:
            return None
        indices = list(img_features_index[synset_id])
        if len(indices) == 1:
            sampled_indices = [0]
        else: 
            sampled_indices = set(np.random.randint(0, len(indices) - 1, num_of_images))
        # idx = random.randint(0, len(indices) - 1)
        img_indices = [indices[idx] for idx in sampled_indices]
        feat_and_boxes = [(img_features[i], img_boxes[i]) for i in img_indices]
        return feat_and_boxes

def choose_image(indices):
    if sum([y for x in indices for y in x]) < 0:
        return None
    sampled_indices = list()
    for indxs in indices:
        sampled_indices.append(np.random.choice(indxs, 1, replace=False)[0])
    return sampled_indices


class MultimodalTxtDataset(Dataset):
    def __init__(
        self,
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: PreTrainedTokenizer,
        crossmod_encoder_tokenizer: PreTrainedTokenizer,
        encoder_name,
        decoder_name,
        crossmod_name,
        start_def_token,
        end_def_token,
        txt_path: str,
        img_features_path: str,
        limit_sentences: int = -1,
        is_infinite=False,
        num_of_images=5,
    ):
        self.num_images_per_synset = num_of_images
        self.examples = list()
        self.is_infinite = is_infinite
        self.encoder_pad_token_id = encoder_tokenizer.pad_token_id
        self.decoder_pad_token_id = decoder_tokenizer.pad_token_id
        self.crossmod_encoder_pad_token_id = crossmod_encoder_tokenizer.pad_token_id
        self.kind2indices = {"txt+img": list(), "txt": list(), "word+img": list(),
                             "img":list()}

        img_features_files = h5py.File(img_features_path, 'r')
        synsets_img_idx = img_features_files["synsets"]
        img_features_index = build_img_features_synsets_index(synsets_img_idx)
        self.img_features = img_features_files["features"]
        self.img_boxes = img_features_files["normalized_boxes"]

        self.txt_only_indices = list()
        self.txt_img_indices = list()
        cache_file_name = (
            txt_path + img_features_path + encoder_name + decoder_name + crossmod_name
        )
        cache_file_name = hashlib.md5(bytes(cache_file_name, "utf8")).hexdigest()
        if os.path.exists(os.path.join(CACHE_DIR, cache_file_name)):
            logger.info("found cached dataset: {}".format(cache_file_name))
            with open(os.path.join(CACHE_DIR, cache_file_name), "rb") as reader:
                self.examples = pkl.load(reader)
                if limit_sentences > 0:
                    self.examples = self.examples[:limit_sentences]
                self.indicize_examples()
                return

        # img_features_files = np.load(img_features_path)
        target_synset_without_bnid = 0
        with open(txt_path) as lines:
            for i, line in enumerate(tqdm(lines)):
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
                img_indices = get_image_indices(
                    target_bnid,
                    img_features_index,
                )
                # prefix = "<sentence&image>"
                ## TODO finish refactoring to include multiple images for each example.
                kind = "txt"
                if img_indices is None:
                    target_synset_without_bnid += 1
                    img_indices = [-1]
                else:
                    kind = "txt+img"
                if id.startswith("word+img"):
                    kind = "word+img"
                elif id.startswith("caption"):
                    kind = "img"
                if kind == "img":
                    sentence = ""
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
                sentence = kind + ": " + sentence
                sentence_ids = encoder_tokenizer.encode(sentence)

                self.examples.append(
                    dict(
                        id=id,
                        input_ids=sentence_ids,
                        input_crossmod_ids=sentence_img_ids,
                        target_bnid=target_bnid,
                        decoder_input_ids=gloss_ids,
                        bnids=bnids,
                        img_indices=img_indices,
                        kind=kind,
                    )
                )
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

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        if self.is_infinite:
            raise NotImplementedError()
        return len(self.examples)

    def get_batch_fun(self):
        def collate_fn(examples):
            (
                ids,
                input_ids,
                input_crossmod_ids,
                img_indices,
                gloss_ids,
                kinds,
            ) = zip(
                *[
                    (
                        e["id"],
                        torch.Tensor(e["input_ids"]).long(),
                        torch.Tensor(e["input_crossmod_ids"]).long(),
                        e["img_indices"],
                        torch.Tensor(e["decoder_input_ids"]).long(),
                        e["kind"],
                    )
                    for e in examples
                ]
            )
            sampled_img_indices = choose_image(img_indices)
            if sampled_img_indices is not None:
                ex_img_features, ex_img_boxes = zip(*[(self.img_features[i], self.img_boxes[i]) for i in sampled_img_indices])
                batched_img_features = torch.Tensor(ex_img_features)
                batched_img_pos = torch.Tensor(ex_img_boxes)
                batched_visual_attention_mask = torch.ones(len(batched_img_features), len(batched_img_features[0]))
            else:
                batched_img_features = torch.Tensor(np.expand_dims(pad_image_features, 0).repeat(len(examples), 0))
                batched_img_pos = torch.Tensor(np.expand_dims(pad_image_features, 0).repeat(len(examples), 0))
                batched_visual_attention_mask = torch.zeros(len(batched_img_features), len(batched_img_features[0]))
            batched_input_ids = pad_sequence(
                input_ids, batch_first=True, padding_value=self.encoder_pad_token_id
            )
            encoder_mask = batched_input_ids != self.encoder_pad_token_id
            token_type_ids = torch.ones(batched_input_ids.shape).long()

            batched_crossmod_input_ids = pad_sequence(
                input_crossmod_ids,
                batch_first=True,
                padding_value=self.crossmod_encoder_pad_token_id,
            )
            crossmod_encoder_mask = (
                batched_crossmod_input_ids != self.crossmod_encoder_pad_token_id
            )
            crossmod_token_type_ids = torch.zeros(
                batched_crossmod_input_ids.shape
            ).long()

            batched_gloss_ids = pad_sequence(
                gloss_ids, batch_first=True, padding_value=self.decoder_pad_token_id
            )
            decoder_input_ids = batched_gloss_ids[:, :-1]
            labels = batched_gloss_ids[:, 1:]
            decoder_mask = decoder_input_ids == self.decoder_pad_token_id
            if kinds is not None:
                assert len(set(kinds)) == 1 or logger.error(
                    "cannot handle batches with example of mixed kinds (txt and txt+img)"
                )
                activate_img_features = "img" in list(set(kinds))[0]
            else:
                activate_img_features = True

            return {
                "ids": ids,
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
            }

        return collate_fn


class ConditionedSampler(Sampler):
    def __init__(
        self,
        indices: Dict[str, List[int]],
        current_index_name: str,
        infinite: bool = True,
    ):
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

