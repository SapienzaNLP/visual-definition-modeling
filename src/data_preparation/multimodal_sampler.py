from random import shuffle
from typing import List

from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, Sampler
from tqdm import tqdm
from transformers import BartTokenizer, LxmertTokenizer


class IndexSampler(Sampler):
    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def shuffle(self):
        shuffle(self.indices)


class SamplerAlternator(BatchSampler):
    def __init__(
        self,
        samplers: List[Sampler],
        batch_size: int,
        drop_last: bool,
        shuffle: bool = False,
        infinite_iterators=False
    ):
        super().__init__(samplers[0], batch_size, drop_last)
        self.samplers = samplers
        self.current_sampler_idx = 0
        self.samplers_size = [len(x) for x in samplers]
        self.shuffle = shuffle
        self.infinite_iterators = infinite_iterators
        self.__init_iterators()

    def __next_sampler(self):
        if len(self.iterators) == 0:
            return None, None
        if self.current_sampler_idx >= len(self.iterators):
            self.current_sampler_idx = 0
        curr_idx = self.current_sampler_idx
        sampler = self.iterators[curr_idx]
        self.current_sampler_idx = (self.current_sampler_idx + 1) % len(self.iterators)
        return sampler, curr_idx

    def __len__(self):
        if self.infinite_iterators:
            raise NotImplementedError()

        tot = sum(len(s) for s in self.samplers)
        if self.drop_last:
            return tot // self.batch_size  # type: ignore
        else:
            return (tot + self.batch_size - 1) // self.batch_size

    def __init_iterators(self):
        if self.shuffle:
            for s in self.samplers:
                s.shuffle()
        self.iterators = [iter(x) for x in self.samplers]

    def __iter__(self):
        self.__init_iterators()
        while True:
            sampler, sampler_idx = self.__next_sampler()
            if sampler is None:  # all samplers are over
                break
            batch = list()
            try:
                for _ in range(self.batch_size):
                    batch.append(next(sampler))
            except Exception:
                if not self.infinite_iterators:
                    del self.iterators[sampler_idx]
                else:
                    self.iterators[sampler_idx] = iter(self.samplers[sampler_idx])
            if len(batch) > 0:
                yield batch
