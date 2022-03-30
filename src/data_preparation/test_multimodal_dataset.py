from torch.utils.data.dataloader import DataLoader
from transformers import LxmertTokenizer

from src.data_preparation.multimodal_datasets import MultimodalTxtDataset

if __name__ == "__main__":
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    dataset = MultimodalTxtDataset(tokenizer, "<d>", "</d>", "data/semcor.data.dm.txt")
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.get_batch_fun())

    i = 0
    for x in data_loader:
        for k, v in x.items():

            print(k, v.shape if v is not None else "None")

        i += 1
        if i % 5:
            break