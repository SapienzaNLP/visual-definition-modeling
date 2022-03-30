from typing import Counter
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from transformers.tokenization_lxmert import LxmertTokenizer
from transformers import BartTokenizer
from tqdm import tqdm

def compute_stats(
    dataset_path,
    img_path
):
    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    encoder_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    decoder_tokenizer = BartTokenizer.from_pretrained(bart_name)

    dataset = MultimodalTxtDataset(
        decoder_tokenizer,
        decoder_tokenizer,
        encoder_tokenizer,
        "<define>",
        "</define>",
        dataset_path,
        img_path,
        limit_sentences=-1,
    )
    kinds = Counter()
    for example in tqdm(dataset.examples, desc="computing stats"):
        kind = example["kind"]
        kinds[kind] += 1
    print(kinds)


if __name__ == "__main__":
    dataset_path="data/in/all.data.dm.txt"
    img_path="data/in/babelpic.frcnn.all_imgs.npz"
    compute_stats(dataset_path, img_path)