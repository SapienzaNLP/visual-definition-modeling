from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_lxmert import LxmertTokenizer


if __name__ == "__main__":
    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    device = "cuda"
    lxmert_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    dev_dataset = MultimodalTxtDataset(
        bart_tokenizer,
        bart_tokenizer,
        lxmert_tokenizer,
        bart_name,
        bart_name,
        lxmert_name,
        "<define>",
        "</define>",
        "data/in/semcor.babelpic_words.dm.new_gloss.txt",
        "data/in/babelpic.mscoco.imgs.h5",
        limit_sentences=-1,
        is_infinite=False,
    )
    samplers = [IndexSampler(dev_dataset.kind2indices[k]) for k in dev_dataset.kind2indices.keys()]

    alternator = SamplerAlternator(
        samplers,
        4,
        drop_last=False,
        shuffle=False,
        infinite_iterators=False

    )
    data_loader = DataLoader(
        dev_dataset,
        batch_sampler=alternator,
        num_workers=12,
        collate_fn=dev_dataset.get_batch_fun(),
    )
    for batch in tqdm(data_loader):
        pass
<<<<<<< HEAD
        
=======
>>>>>>> 29a7d41ae08c8ea0aecbc97cea245b7800194aa8
