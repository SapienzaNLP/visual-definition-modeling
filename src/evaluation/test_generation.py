import os

import _pickle as pkl
import torch
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.modeling.model_full_bart_visual_feat import MGMFullBart
from src.training.model_full_bart_trainer import (MGMFinetuner,
                                                  TrainingParams,
                                                  get_training_params)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BartTokenizer, LxmertTokenizer


def adapt_batch(self, batch, precision, device):
        for k, v in batch.items():
            if type(v) is not torch.Tensor:
                continue
            v = v.to(device)
            # if precision == 16 and v.dtype == torch.float:
            #     v = v.half()
            batch[k] = v
        return batch

if __name__ == "__main__":
    device = "cpu"
    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"

    chpt_path= "/home/npf290/dev/MultimodalGlosses/data/out/pretraining-plain-no-task-warmup/wandb/run-20210509_175606-1rpjnhju/files/checkpoints/global_step=9999.ckpt"
    checkpoint = torch.load(chpt_path, map_location=torch.device(device))
    lxmert_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    out_file = os.path.join('/'.join(chpt_path.split('/')[:-1]),"debug_generation.word+img.{}.txt".format(chpt_path.split('/')[-4]))
    print("Out file:", out_file)
    batch_size = 1

    id2gold = dict()
    with open(
            'data/in/imagenet.islvrc.train.words.dm.gold.key.txt',
            'rt') as lines:
        for line in lines:
            fields = line.rstrip().split()
            id2gold[fields[0]] = fields[1:]

    training_params = get_training_params()
    training_params["batch_size"] = batch_size
    model = MGMFullBart.from_pretrained(bart_name, multimodal_encoder=lxmert_name, **training_params)

    state_dict = checkpoint["state_dict"]

    for k, v in list(state_dict.items()):
        del state_dict[k]
        state_dict[k.replace("lxmertlm.", "")] = v

    model.load_state_dict(state_dict)
    model.to(device)

    dataset = MultimodalTxtDataset(bart_tokenizer,
                                         bart_tokenizer,
                                         lxmert_tokenizer,
                                         bart_name, bart_name, lxmert_name,
                                         "<define>", "</define>",
                                         "data/in/all.data.dm.new_gloss.txt",
                                         "babelpic.frcnn.all_imgs.npz",
                                         limit_sentences=10)

    data_loader = DataLoader(dataset, batch_size=1,
                             collate_fn=dataset.get_batch_fun())
    # tokenizer = decoder_tokenizer
    generation_params = dict(num_beams=4,
                             min_length=5,
                             max_length=25,
                             temperature=1,
                             repetition_penalty=2,
                             length_penalty=1.5,
                             no_repeat_ngram_size=2)
    num_beams = generation_params["num_beams"]
    lxmert_lm = lxmert_lm.eval()
    writer_oth = open(out_file, 'wt')
    with_imgs_writer = open(out_file + ".with_imgs", 'wt')
    no_imgs_writer = open(out_file + ".no_imgs", "wt")
    with torch.no_grad():
        for x in tqdm(data_loader, desc="Generating"):
            sent_id = x['ids']
            labels = x["labels"]
            input_len = x["input_ids"].shape[-1]
            no_images = torch.sum(x["visual_feats"]).item() == 0

            beam_output = lxmert_lm.generate(
                    x["input_ids"].to(device),
                    **generation_params,
                    decoder_start_token_id=bart_tokenizer.bos_token_id,
                    decoder_kwargs={},
                    encoder_kwargs={"crossmod_input_ids": x["crossmod_input_ids"].to(device),
                                    "crossmod_attention_mask": x["crossmod_attention_mask"].to(device),
                                    "crossmod_token_type_ids": x["crossmod_token_type_ids"].to(device),
                                    "batch_kinds": x["batch_kinds"],
                                    "visual_feats": x["visual_feats"].to(device),
                                    "visual_pos": x["visual_pos"].to(device),
                                    "visual_attention_mask": x["visual_attention_mask"].to(device),
                                    "activate_img_features": x["activate_img_features"]}
                )

            print("Output:\n" + 100 * '-')
            print(bart_tokenizer.decode(x["input_ids"][0]))
            print(bart_tokenizer.decode(beam_output[0]))
            print(bart_tokenizer.decode(labels[0]).strip())
            writer = no_imgs_writer if no_images else with_imgs_writer
            writer.write('{}\t{}\n{}\n{}\n\n'.format(sent_id[0], '\t'.join(id2gold[sent_id[0]]),
                                                         bart_tokenizer.decode(labels[0]).strip(),
                                                         bart_tokenizer.decode(beam_output[0][1:-1].strip())
                                                         ))
            writer_oth.write('{}\t{}\n'.format(sent_id[0], bart_tokenizer.decode(beam_output[0][1:-1])).strip())
            writer.flush()
            print()
    no_imgs_writer.close()
    with_imgs_writer.close()
