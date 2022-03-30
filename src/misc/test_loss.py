from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import LxmertTokenizer, BartTokenizer

from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.modeling.model import Lxmert_LM
import _pickle as pkl
import torch

from src.training.model_trainer import TrainingParams, LxmertLMFinetuner, get_training_params

if __name__ == "__main__":
    device = "cpu"
    # chpt_path = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data/out/wandb/latest-run/files/checkpoints/epoch=8.ckpt"
    chpt_path = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data_ssd4/out/wandb/latest-run/files/checkpoints/epoch=27.ckpt"
    checkpoint = torch.load(chpt_path, map_location=torch.device(device))
    encoder_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    out_file = "data/out/debug_generation.training.txt"
    batch_size = 1

    id2gold = dict()
    with open(
            '/home/tommaso/Documents/data/WSD_Evaluation_Framework_3.0/Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt',
            # '/home/tommaso/Documents/data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt',
            'rt') as lines:
        for line in lines:
            fields = line.rstrip().split()
            id2gold[fields[0]] = fields[1:]

    training_params = get_training_params()
    training_params["batch_size"] = 1
    # tuner = LxmertLMFinetuner.load_from_checkpoint(chpt_path, model, training_params)
    # lxmert_lm = tuner.lxmertlm
    lxmert_lm = Lxmert_LM()

    state_dict = checkpoint["state_dict"]

    for k, v in list(state_dict.items()):
        del state_dict[k]
        state_dict[k.replace("lxmertlm.", "")] = v

    lxmert_lm.load_state_dict(state_dict)
    lxmert_lm = lxmert_lm.cpu()

    dataset = MultimodalTxtDataset(encoder_tokenizer,
                                   decoder_tokenizer,
                                   "<d>", "</d>",
                                   "data/in/semeval2007.data.dm.txt",
                                   # "data/in/semcor.data.dm.all.txt",
                                   "data/in/babelpic.frcnn.semcor.all.npz")

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=dataset.get_batch_fun())
    # tokenizer = decoder_tokenizer
    generation_params = dict(num_beams=4,
                             min_length=5,
                             max_length=15,
                             temperature=1,
                             repetition_penalty=2,
                             length_penalty=1.5,
                             no_repeat_ngram_size=2)
    num_beams = generation_params["num_beams"]
    lxmert_lm = lxmert_lm.eval()
    losses = list()
    with torch.no_grad():
        for x in tqdm(data_loader):
            # sent_id = x['ids']
            # labels = x["labels"]
            # input_len = x["input_ids"].shape[-1]
            # x["input_ids"] = x["input_ids"].to(device)
            # for k, v in x.items():
            #     if k == "input_ids" or k == 'ids' or k == 'batch_kinds':
            #         continue
            #     x[k] = v.to(device)
            # decoder_input_ids = torch.Tensor([decoder_tokenizer.bos_token_id]).unsqueeze(0).long().to(device)
            # no_images = torch.sum(x["visual_feats"]).item() == 0
            outputs = lxmert_lm(**x)
            val_loss = outputs[0]
            losses.append(val_loss)
    print(sum(losses)/len(losses))
