from src.data_preparation.multimodal_datasets_lazy import CircularSampler, DatasetAlternator, ModalitySampler
from src.data_preparation.multimodal_datasets_lazy import FolderDataset
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_bart import BartTokenizer
from transformers import LxmertTokenizer


if __name__ == '__main__':
    img_dataset_path = "data/in/training/img/"
    txt_dataset_path = "data/in/training/txt"
    txt_img_dataset_path = "data/in/training/txt+img"
    word_img_dataset_path = "data/in/training/word+img"
    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    batch_size = 8
    img_dataset = FolderDataset(img_dataset_path, shuffle=True)
    txt_dataset = FolderDataset(txt_dataset_path, shuffle=True)
    txt_img_dataset = FolderDataset(txt_img_dataset_path, shuffle=True)
    word_img_dataset = FolderDataset(word_img_dataset_path, shuffle=True)
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    lxmert_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    lxmert_pad_token_id = lxmert_tokenizer.pad_token_id
    bart_pad_token_id = bart_tokenizer.pad_token_id
    dataset = DatasetAlternator(
        lxmert_pad_token_id,
        bart_pad_token_id,
        bart_pad_token_id,
        img_dataset,
        txt_dataset,
        txt_img_dataset,
        word_img_dataset,
    )
    modality_batch_sampler = ModalitySampler(CircularSampler(4), batch_size)
    data_loader = DataLoader(
        dataset,
        batch_sampler=modality_batch_sampler,
        num_workers=0,
        collate_fn=dataset.get_batch_fun(),
    )
    for batch in data_loader:
        if batch['batch_kinds'][0] == 'word+img':
            print(set(batch["batch_kinds"]))