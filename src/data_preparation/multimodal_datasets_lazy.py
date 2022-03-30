from queue import Empty
import time
import numpy as np
import itertools
import os
import json
import logging
import pickle as pkl
import random
from typing import Counter, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler
from transformers.tokenization_bart import BartTokenizer, BartTokenizerFast
from transformers import LxmertTokenizer
import multiprocessing
import glob

random.seed(34)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

pad_image_features = np.zeros((36, 2048))
pad_image_boxes = np.zeros((36, 4))

def get_batch_fun(encoder_pad_token_id, decoder_pad_token_id, img_source):
    def collate_fn(examples):
        if type(examples[0]) == list: # to handle dev iterator
            examples = examples[0]
        (
            ids,
            input_ids,
            lxmert_feats,
            gloss_ids,
            kinds,
            indexes,
            visual_pos,
            frcnn_feats
        ) = zip(
            *[
                (
                    e["id"],
                    torch.Tensor(e["input_ids"]).long(),
                    torch.Tensor(e["lxmert_feats"])
                    if e["lxmert_feats"] is not None
                    else None,
                    torch.Tensor(e["decoder_input_ids"]).long(),
                    e["kind"],
                    e["index"],
                    torch.Tensor(e["visual_pos"]) if e['visual_pos'] is not None else None,
                    torch.Tensor(e["visual_feats"]) if e['visual_feats'] is not None else None
                )
                for e in examples
            ]
        )
        if img_source == 'lxmert': 
            visual_feats = lxmert_feats
        elif img_source == 'frcnn':
            visual_feats = frcnn_feats
        else:
            print(f'[ERROR] cannot recognise img_source {img_source}.')
            raise RuntimeError(f'Invalid img_source {img_source}.')
        if visual_feats[0] is None:
            assert all(x is None for x in visual_feats)
            visual_feats, visual_pos = None, None
        elif len(visual_feats[0].shape) == 2:
                visual_feats = [vf.unsqueeze(0) for vf in visual_feats]
                visual_pos = [vp.unsqueeze(0) for vp in visual_pos]
            
        if visual_feats is not None:
           
        
            # if len(visual_feats[0].squeeze().shape) > 2: # then more than one image for a single example, thus let's sample one
            rand_indexes = [np.random.randint(len(e)) for e in visual_feats]
            batched_visual_feats = torch.stack([e[i] for i,e in zip(rand_indexes, visual_feats)], 0)
            batched_visual_pos = torch.stack([e[i] for i, e in zip(rand_indexes, visual_pos)], 0)
            # else: 
            #     visual_feats = [vf.squeeze() for vf in visual_feats]
            #     visual_pos = [vp.squeeze() for vp in visual_pos]
            #     batched_visual_feats = torch.stack(visual_feats, 0)
            #       batched_visual_pos = torch.stack(visual_pos, 0)
            batched_visual_attention_mask = torch.ones(
                len(batched_visual_feats), len(batched_visual_feats[0])
            )
        else:
            batched_visual_feats = torch.Tensor(
                np.expand_dims(pad_image_features, 0).repeat(
                    len(examples), 0)
            )
            batched_visual_pos = torch.Tensor(
                np.expand_dims(pad_image_features, 0).repeat(
                    len(examples), 0)
            )
            batched_visual_attention_mask = torch.zeros(
                len(batched_visual_feats), len(batched_visual_feats[0])
            )

        # if ex_img_features is not None:
        #     if len(ex_img_features[0].shape) > 2:
        #         batched_img_features = torch.stack([e[np.random.randint(len(e))] for e in ex_img_features], 0)
        #     else: 
        #         batched_img_features = torch.stack(ex_img_features, 0)
        #     batched_visual_attention_mask = torch.ones(
        #         len(batched_img_features), len(batched_img_features[0])
        #     )
        # else:
        #     batched_img_features = torch.Tensor(
        #         np.expand_dims(pad_image_features, 0).repeat(
        #             len(examples), 0)
        #     )
        #     batched_visual_attention_mask = torch.zeros(
        #         len(batched_img_features), len(batched_img_features[0])
        #     )
            
        batched_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=encoder_pad_token_id
        )
        encoder_mask = batched_input_ids != encoder_pad_token_id
        token_type_ids = torch.ones(batched_input_ids.shape).long()

        batched_gloss_ids = pad_sequence(
            gloss_ids, batch_first=True, padding_value=decoder_pad_token_id
        )
        decoder_input_ids = batched_gloss_ids[:, :-1]
        labels = batched_gloss_ids[:, 1:]
        decoder_mask = decoder_input_ids == decoder_pad_token_id
        if kinds is not None:
            assert len(set(kinds)) == 1 or logger.error(
                "cannot handle batches with example of mixed kinds (txt and txt+img)"
            )
            activate_img_features = "img" in list(set(kinds))[0] or "q&a" in list(set(kinds))[0]
        else:
            activate_img_features = True

        insert_img_object = "img+img" in list(set(kinds))[0]

        return {
            "ids": ids,
            "input_ids": batched_input_ids,
            "attention_mask": encoder_mask,
            "token_type_ids": token_type_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_padding_mask": decoder_mask,
            "visual_attention_mask": batched_visual_attention_mask,
            "labels": labels.contiguous(),
            "batch_kinds": kinds,
            "activate_img_features": activate_img_features,
            "insert_img_objects": insert_img_object,
            "target_indexes": indexes,
            "visual_feats": batched_visual_feats,
            "visual_pos": batched_visual_pos
        }

    return collate_fn

class CircularSampler(Sampler):
    def __init__(self, ln) -> None:
        self.ln = ln

    def __iter__(self):
        return itertools.cycle(range(self.ln))


class CircularSamplerWithCoherentAccumulation(CircularSampler):
    def __init__(self, ln, accumulation_step) -> None:
        self.ln = ln
        self.accumulation_step = accumulation_step

    def __iter__(self):
        lst = list()
        for i in range(self.ln):
            lst.extend([i] * self.accumulation_step)
        return itertools.cycle(lst)


def linear_convergence_function(
    current_step, start_step, start_prob, target_step, target_prob
):
    if current_step < start_step:
        return start_prob
    if current_step > target_step:
        return target_prob

    current_prob = (
        (current_step - start_step)
        / (target_step - start_step)
        * (target_prob - start_prob)
    ) + start_prob
    return current_prob


def exponential_convergence_function(
    current_step, start_step, start_prob, target_step, target_prob
):
    if start_step != 0:
        raise RuntimeError("Cannot compute the function if start_step != 0")
    if current_step < start_step:

        return start_prob
    current_prob = (
        (start_prob * (target_prob / start_prob) ** (1.0 / target_step))
    ) ** current_step
    return current_prob


class CurriculumSampler(CircularSampler):
    def __init__(
        self,
        num_tasks: int,
        batch_size: int,
        gradient_accumulation: int,
        convergence_function: str,
        initial_probabilities: dict,
        target_probabilities: dict,
        # how many steps to wait before start increasing / decreasing task probability
        tasks_waiting_steps: dict,
        # how many steps we have to take to bring task probability from initial to target
        tasks_target_steps: dict,
        train_bart_only_steps:int,
        train_img_only_steps:int,
    ) -> None:
        """[summary]
        Args:
            num_tasks (int): number of tasks to combine.
            batch_size (int): size of the batch that will be used.
            gradient_accumulation (int): number of steps gradient is accumulated.
            convergence_function (str): name of the function to use to bring initial probabilities to target probabilities.
            initial_probabilities (dict): dictionary with sampling probabilities for each task.
            target_probabilities (dict): dictionary with sampling probabilities for each task once cv is unrolled.
            tasks_waiting_steps (dict): dictionary with number of steps we have to wait befor starting to change the probability of each task.
            tasks_target_steps (dict): dictionary with number of steps it is going to take to transform the task initial probability to its target probability.
        """
        self.num_tasks = num_tasks
        self.gradient_accumulation = gradient_accumulation
        self.convergence_function_name = convergence_function
        self.tasks = list(initial_probabilities.keys())
        self.tasks_indices = list(range(num_tasks))
        self.tasks_waiting_steps = [tasks_waiting_steps[t] for t in self.tasks]
        self.tasks_target_steps = [tasks_target_steps[t] for t in self.tasks]
        self.initial_probabilities = [
            initial_probabilities[t] for t in self.tasks]
        self.target_probabilities = [
            target_probabilities[t] for t in self.tasks]
        #self.current_probabilities = list(self.initial_probabilities)
        self.current_step = 0
        self.batch_size = batch_size
        self.train_bart_only_steps = train_bart_only_steps
        self.train_img_only_steps = train_img_only_steps
        self.img_only_probabilities = [1.0 if t == 'img+img' else 0. for t in self.tasks]
        self.current_probabilities = [1.0 if t == 'txt' else 0. for t in self.tasks]
        self.convergence_function = self.__get_convergence_function(
            convergence_function
        )
        for i, task in enumerate(self.tasks):
            print(task, "start_prob", self.initial_probabilities[i], "end_prob", self.target_probabilities[i],
                  "steps_to_wait", self.tasks_waiting_steps[i], "steps_target", self.tasks_target_steps[i])

    def __get_convergence_function(self, name):
        if name == "linear":
            return linear_convergence_function
        if name == "exp":
            return exponential_convergence_function
        raise RuntimeError(
            name +
            " convergence function unknown please select one among [linear, exp]"
        )

    def update_probabilities(self):
        numb_batch_saw = self.current_step
        training_steps = numb_batch_saw // self.gradient_accumulation
        # if training_steps > 50:
        # print()
        s = 0
        for i in range(self.num_tasks):
            current_task_prob = self.convergence_function(
                training_steps,
                self.tasks_waiting_steps[i],
                self.initial_probabilities[i],
                self.tasks_target_steps[i],
                self.target_probabilities[i],
            )
            self.current_probabilities[i] = current_task_prob
            s += current_task_prob
        self.current_probabilities = [x/s for x in self.current_probabilities]
        # print(self.current_probabilities)

    def __iter__(self):
        joint_training = False
        while True:
            idx = np.random.choice(
                self.tasks_indices, 1, False, self.current_probabilities
            )[0]
            self.current_step += 1
            training_step = self.current_step // self.gradient_accumulation
            if not joint_training and training_step == self.train_bart_only_steps:
                self.current_probabilities = self.img_only_probabilities
            if training_step == self.train_img_only_steps and not joint_training:
                joint_training = True
                self.current_probabilities = self.initial_probabilities
                self.current_step = 0
            if joint_training:
                self.update_probabilities()
            # print(self.current_step, self.current_probabilities)
            yield idx


class ModalitySampler(BatchSampler):
    def __init__(
        self, sampler, batch_size, drop_last= False
    ) -> None:
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        for modality in self.sampler:
            yield [modality] * self.batch_size


class DatasetAlternator(Dataset):
    def __init__(
        self,
        encoder_pad_token_id,
        decoder_pad_token_id,
        img_source,
        # max_len = 300,
        *datasets
    ) -> None:
        super().__init__()
        self.encoder_pad_token_id = encoder_pad_token_id
        self.decoder_pad_token_id = decoder_pad_token_id
        self.dataset_iterators = [iter(x) for x in datasets]
        self.datasets = datasets
        self.num_datasets = len(self.datasets)
        self.img_source = img_source
        # self.max_len = max_len
        self.shuffle = False

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, index):
        try:
            return next(self.dataset_iterators[index])
        except StopIteration:
            self.dataset_iterators[index] = iter(self.datasets[index])
            return next(self.dataset_iterators[index])
    
    def reset(self):
        for d in self.datasets:
            d.stop()
        self.dataset_iterators = [iter(x) for x in self.datasets]

        
    def get_batch_fun(self):
        return get_batch_fun(self.encoder_pad_token_id, self.decoder_pad_token_id, self.img_source)


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as reader:
        examples = pkl.load(reader)
    return examples

def produce(folder, shuffle, queue, max_len, is_infinite, stop):
    files = [f for f in os.listdir(folder) if f.endswith(".pkl")][:]
    if shuffle:
        random.shuffle(files)
    i = 0
    while True:
        with stop.get_lock():
            if stop.value == True:
                break
        pkl_file = files[i]
        examples = load_pkl(os.path.join(folder, pkl_file))

        if shuffle:
            random.shuffle(examples)
        for ex in examples:
            if max_len is not None and len(ex["input_ids"]) > max_len:
                continue
            # if ex['visual_feats'] is not None and ex['lxmert_feats'] is None:
                # continue
            with stop.get_lock():
                if stop.value == True:
                    break
            ex['file'] = pkl_file
            queue.put(ex)
        i += 1
        if i == len(files):
            if is_infinite:
                if shuffle:
                    random.shuffle(files)
                i = 0
            else:
                queue.put(-1)
                with stop.get_lock():
                    stop.value = True


class FiniteFolderDataset(Dataset):
    def __init__(self, folder_path, name, img_source, max_len=None, shuffle=False, encoder_pad_token_id=None,
    decoder_pad_token_id=None) -> None:
        super().__init__()
        self.root = folder_path
        self.name = name
        with open(os.path.join(self.root, "metadata.json")) as reader:
            self.metadata = json.load(reader)
        self.pkl_files = [os.path.join(self.root, f) for f in os.listdir(
            folder_path) if f.endswith("pkl")]
        if shuffle:
            shuffle(self.pkl_files)
        self.num_files = len(self.pkl_files)
        self.shuffle = shuffle
        self.queue_size = 7500
        self.examples_to_yield = []#multiprocessing.Queue(self.queue_size)
        self.queue_size = multiprocessing.Value('i', 0)
        self.max_len = max_len
        self.encoder_pad_token_id = encoder_pad_token_id
        self.decoder_pad_token_id = decoder_pad_token_id
        self.paths_iterator = iter(self.pkl_files)
        self.steps = 0
        self.img_source = img_source

    def get_batch_fun(self):
        return get_batch_fun(self.encoder_pad_token_id, self.decoder_pad_token_id, self.img_source)

    def fill_queue(self, pkl_path):
        with open(pkl_path, 'rb') as reader:
            data = pkl.load(reader)
        if self.shuffle:
            random.shuffle(data)
        self.examples_to_yield.extend(data)
    def __len__(self):
        return self.metadata['examples']


    def __getitem__(self, index):
        if len(self.examples_to_yield) == 0:
            path = next(self.paths_iterator)
            # process = multiprocessing.Process(FiniteFolderDataset.fill_queue, args=[self.examples_to_yield, path, self.queue_size])
            self.fill_queue(path)
                
                # process.start()
        
        ex = self.examples_to_yield.pop()
        self.steps += 1
        if self.steps == len(self):
            if self.shuffle:
                random.shuffle(self.pkl_files)
            self.paths_iterator = iter(self.pkl_files)
            self.steps = 0
        return ex

class FolderDataset(IterableDataset):
    def __init__(
        self,
        folder_path,
        name,
        img_source,
        max_len=None,
        shuffle=False,
        is_infinite=False,
        encoder_pad_token_id=None,
        decoder_pad_token_id=None,
    ) -> None:
        super().__init__()
        self.root = folder_path
        self.name = name
        with open(os.path.join(self.root, "metadata.json")) as reader:
            self.metadata = json.load(reader)
        self.pkl_files = [f for f in os.listdir(
            folder_path) if f.endswith("pkl")]
        self.num_files = len(self.pkl_files)
        self.shuffle = shuffle
        self.queue_size = 7500
        self.examples_to_yield = multiprocessing.Queue(self.queue_size)
        self.max_len = max_len
        self.is_infinite = is_infinite
        self.stop_producer = multiprocessing.Value('i', False)
        self.producer = None
        self.producer_is_running = False
        self.encoder_pad_token_id = encoder_pad_token_id
        self.decoder_pad_token_id = decoder_pad_token_id
        self.img_source = img_source
    
    def start_producer(self):
        with self.stop_producer.get_lock():
            self.stop_producer.value = False
        producer_process = multiprocessing.Process(
            target=produce, args=[self.root, self.shuffle, self.examples_to_yield, self.max_len, self.is_infinite, self.stop_producer]
        )
        producer_process.start()
        self.producer_is_running = True
        return producer_process
    
    def stop(self):
        if self.producer is None:
            return
        with self.stop_producer.get_lock():
            self.stop_producer.value = True
        try:
            while True:
                self.examples_to_yield.get(block=False)
        except Empty:
            pass
        self.examples_to_yield.close()
        self.producer.terminate()
        self.producer.join()
        self.examples_to_yield = multiprocessing.Queue(self.queue_size)


    def __iter__(self):
        try:
            """
            if i just iterates for 10 examples (e.g., when we generate at the end of an epoch) this will then continue from that point
            (in validation) hence validating on less examples (this does not happen because in pretraining we use another class for the validation
            set). This happens when fintuning for vqa where we also have a huge validation set and thus we need this dataset
            """
            self.producer = self.start_producer()
            i = 0
            while True:
                ex = self.examples_to_yield.get()
                i += 1
                if ex == -1:
                    self.stop()
                    break
                yield ex
        except KeyboardInterrupt as e:
            print(e)
            print("terminating")
            self.stop()

    def get_batch_fun(self):
        return get_batch_fun(self.encoder_pad_token_id, self.decoder_pad_token_id, self.img_source)

class PickleDataset(Dataset):
    def __init__(self, paths:Union[str,List], pad_token_id, batch_size, img_source) -> None:
        super().__init__()
        if type(paths) is str:
            paths = [paths]
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.img_source  = img_source
        self.batches = list()
        for path in paths:
            data = load_pkl(path)
            for i in range(0, len(data), batch_size):
                self.batches.append(data[i:i+batch_size])
            
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]
    
    def get_batch_fun(self):
        return get_batch_fun(self.pad_token_id, self.pad_token_id, self.img_source)

class PicklesDataset(Dataset):
    def __init__(self, root, pad_token_id, batch_size, img_source) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.img_source = img_source
        self.batches = list()
        # load all pikles in the root and subfolders
        for file in glob.glob(os.path.join(self.root, '*.pkl')):
            data = load_pkl(file)
            for i in range(0, len(data), batch_size):
                self.batches.append(data[i:i+batch_size])
        for file in glob.glob(os.path.join(self.root, '*', '*.pkl')):
            data = load_pkl(file)
            for i in range(0, len(data), batch_size):
                self.batches.append(data[i:i+batch_size])
            
    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]
    
    def get_batch_fun(self):
        return get_batch_fun(self.pad_token_id, self.pad_token_id)

    def __getitem__(self, index):
        return self.batches[index]
    
    def get_batch_fun(self):
        return get_batch_fun(self.pad_token_id, self.pad_token_id, self.img_source)

if __name__ == '__main__':
    bart_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
    batch_size = 4
    dataset = PicklesDataset('/root/MultimodalGlosses/data/in/dev_cap', bart_tokenizer.pad_token_id, bart_tokenizer.pad_token_id, batch_size)
    dataloader = DataLoader(dataset, 1, collate_fn=dataset.get_batch_fun())
    counter = Counter()
    for batch in dataloader:
        kind =list(set(batch['batch_kinds']))[0]
        counter[kind]+= 1
    print(counter)
