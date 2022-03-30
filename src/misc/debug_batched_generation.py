import os
from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator

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

def generate(x):
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
    return beam_output

if __name__ == "__main__":
    device = "cuda"
    bart_name = "facebook/bart-large"
    lxmert_name = "unc-nlp/lxmert-base-uncased"
    # chpt_path = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data/out/wandb/latest-run/files/checkpoints/epoch=8.ckpt"

    chpt_path= "data/out/wandb/run-20210111_184023-1c7umlxa-lxmert-bart-linear-txt-txt+img-word+img-captioning-curriculum-learning/files/checkpoints/global_step=24999.ckpt"
    checkpoint = torch.load(chpt_path, map_location=torch.device(device))
    lxmert_tokenizer = LxmertTokenizer.from_pretrained(lxmert_name)
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    batch_size = 16

    training_params = get_training_params()
    training_params["batch_size"] = batch_size
    lxmert_lm = MGMFullBart.from_pretrained(bart_name, multimodal_encoder=lxmert_name, **training_params)

    state_dict = checkpoint["state_dict"]

    for k, v in list(state_dict.items()):
        del state_dict[k]
        state_dict[k.replace("lxmertlm.", "")] = v

    lxmert_lm.load_state_dict(state_dict)
    lxmert_lm.to(device)

    dataset = MultimodalTxtDataset(bart_tokenizer,
                                         bart_tokenizer,
                                         lxmert_tokenizer,
                                         bart_name, bart_name, lxmert_name,
                                         "<define>", "</define>",
                                         "data/in/all.data.dm.new_gloss.txt",
                                         "data/in/babelpic.frcnn.all_imgs.npz",
                                         limit_sentences=-1)

    samplers = [
            IndexSampler(dataset.kind2indices[k])
            for k in dataset.kind2indices.keys()
        ]

    alternator = SamplerAlternator(
        samplers,
        batch_size,
        drop_last=False,
        shuffle=False,
        infinite_iterators=False,
    )
    data_loader = DataLoader(
        dataset,
        batch_sampler=alternator,
        num_workers=0,
        collate_fn=dataset.get_batch_fun(),
    )

    # tokenizer = decoder_tokenizer
    generation_params = dict(num_beams=4,
                             min_length=1,
                             max_length=512,
                             temperature=1,
                             repetition_penalty=2,
                             length_penalty=1.5,
                             no_repeat_ngram_size=2,
                             num_return_sequences=4)
    num_beams = generation_params["num_beams"]
    lxmert_lm = lxmert_lm.eval()
    with torch.no_grad():
        for x in data_loader:
            sent_id = x['ids']
            labels = x["labels"]
            input_len = x["input_ids"].shape[-1]
            no_images = torch.sum(x["visual_feats"]).item() == 0
            batched_beam_output = generate(x)
            single_beam_output = list()
            for i in range(batch_size):
                single_ex = dict()
                for k, v in x.items():
                    if k == "activate_img_features":
                        single_ex[k] = v
                    else:
                        single_ex[k] = v[i].unsqueeze(0) if type(v[i]) == torch.Tensor else v[i]
                o = generate(single_ex)
                single_beam_output.append(o)
            print("\n\n\n\n\n")
            print("\n".join([bart_tokenizer.decode(b, skip_special_tokens=True) for b in x["input_ids"]]))
            print("\n".join([bart_tokenizer.decode(l, skip_special_tokens=True) for l in labels]))

            print("Output - Batched:\n" + 100 * '-')
            print("\n".join([bart_tokenizer.decode(b, skip_special_tokens=True) for b in batched_beam_output]))

            print("Output - Single:\n" + 100 * '-')
            print("\n".join([bart_tokenizer.decode(b[0], skip_special_tokens=True) for b in single_beam_output]))
            print("="*40)
