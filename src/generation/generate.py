from src.finetuning.finetuning import load_model
from src.data_preparation.multimodal_datasets_lazy import DatasetAlternator, FiniteFolderDataset, FolderDataset, PickleDataset
from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator
from transformers import AutoTokenizer
from argparse import ArgumentParser
from src.training.model_full_bart_trainer import MGMFinetuner, get_training_params
from transformers.tokenization_lxmert import LxmertTokenizer
from src.modeling.components import prepare_encoder_input
from src.modeling.model_full_bart_visual_feat import MGMFullBart

from torch.utils.data.dataloader import DataLoader
from src.data_preparation.multimodal_datasets import (
    MultimodalTxtDataset,
)
import json
import torch
from tqdm import tqdm
from transformers.tokenization_bart import BartTokenizer
from sentence_transformers import SentenceTransformer
from src.commons.commons import move_batch
from pprint import pprint
import re


def generate(
    model,
    dataset,
    beam_size,
    num_return_sequences,
    tokenizer,
    normalize_sentence_probs,
    uniform_sentence_probs,
    device="cuda",
    **kwargs
):
    bos_token_id=tokenizer.bos_token_id
    generation_params = dict(
        num_beams=beam_size,
        min_length=kwargs.get('min_length', 1),
        max_length=kwargs.get('max_length', 512),
        temperature=1,
        repetition_penalty=2,
        length_penalty=1,
        no_repeat_ngram_size=2,
        num_return_sequences=num_return_sequences,
    )
    print('-'* 100)
    print('Generation Params:')
    print('-'* 100)
    print('\n'.join([f'{k}: {v}' for k, v in generation_params.items()]))
    print('-'* 100)
    outputs = list()
    with torch.no_grad():
        for x in tqdm(dataset, desc="Generating"):
            sent_id = x["ids"]
            x = move_batch(x, device)
            encoder_inputs = prepare_encoder_input(x)
            beam_output = model.generate(
                x["input_ids"],
                **generation_params,
                decoder_start_token_id=bos_token_id,
                decoder_kwargs={},
                encoder_kwargs=encoder_inputs
            )
            # beam_output = beam_output.view(-1, num_return_sequences, beam_output.size(-1))
            scored_generations = score_generations(
                model,
                x,
                beam_output,
                tokenizer,
                num_output_sequences=num_return_sequences,
                normalize_probs=normalize_sentence_probs,
                uniform_probs=uniform_sentence_probs,
            )
            for i in range(len(x["ids"])):
                outputs.append(
                    {
                        "id": x["ids"][i],
                        "input_sentence": tokenizer.decode(
                            x["input_ids"][i], skip_special_tokens=True
                        ),
                        "generations": scored_generations[i],
                    }
                )

    return outputs


def get_uniform_prob_output(beam_output, decoded_output):
    up = 1 / len(beam_output)
    generations = list()
    for o, bo in zip(decoded_output, beam_output):
        x = {
            "output": o,
            "encoded_output": bo.detach().cpu(),
            "sentence_probability": up,
        }
        generations.append(x)
    return generations


def score_generations(
    model, x, beam_output, tokenizer, normalize_probs, num_output_sequences, uniform_probs=False
):
    decoded_output = [
        tokenizer.decode(o if type(o)==list else o.tolist(), skip_special_tokens=True) for o in beam_output
    ]
    if uniform_probs:
        return get_uniform_prob_output()
    seq_len = x["input_ids"].size(-1)
    
    outputs = model(
        input_ids=x["input_ids"].unsqueeze(1).repeat(
            1, num_output_sequences, 1).view(-1, seq_len),
        decoder_input_ids=beam_output,
        attention_mask=x["attention_mask"].unsqueeze(1).repeat(
            1, num_output_sequences, 1).view(-1, seq_len),
        batch_kinds=x["batch_kinds"] * num_output_sequences,
        visual_feats=x["visual_feats"].unsqueeze(1).repeat(
            1, num_output_sequences, 1, 1).view(-1, *x["visual_feats"].shape[1:]),
        visual_attention_mask=x["visual_attention_mask"].unsqueeze(
            1).repeat(1, num_output_sequences, 1).view(-1, x["visual_attention_mask"].size(-1)),
        activate_img_features=x["activate_img_features"],
        insert_img_objects=x["insert_img_objects"],
        target_indexes=x["target_indexes"] * num_output_sequences,
        return_dict=True,
        use_cache=False,
    )
    beam_output = beam_output.cpu()
    beam_output = torch.cat(
        [beam_output[:, 1:], torch.ones(beam_output.size(0), 1).long()], -1
    )
    logits = outputs["logits"].cpu()
    probs = torch.softmax(logits, -1)
    token_probs = probs[
        torch.arange(0, probs.size(0)).long().unsqueeze(1),
        torch.arange(0, probs.size(1)).long().unsqueeze(0),
        beam_output,
    ]
    beam_mask = beam_output == 1
    token_probs.masked_fill_(
        beam_mask, 1
    )  # fills with 1s as log(1) = 0 and thus padding won't be taken into account for the sentence score
    sentence_prob = torch.prod(token_probs, -1)
    generations = list()
    for o, bo, sp in zip(decoded_output, beam_output, sentence_prob):
        x = {
            "output": o,
            "encoded_output": bo.detach().cpu(),
            "sentence_probability": sp.item(),
        }
        generations.append(x)
    if normalize_probs:
        den = sum(g["sentence_probability"] for g in generations)
        for g in generations:
            g["sentence_probability"] = g["sentence_probability"] / den
    grouped_generations = list()
    for i in range(0, len(generations), num_output_sequences):
        grouped_generations.append(generations[i:i+num_output_sequences])
    return grouped_generations


TEST_MAPPING={
        # "captioning":["data/in/test_cap/img/pkl_0.pkl"],
        "word+img":["data/in/bart-tokenized/test/word+img/"],
        # "wsd":["data/in/test_cap/txt/pkl_0.pkl", "data/in/test_cap/txt+img/pkl_0.pkl"],
        "img+img": ["data/in/bart-tokenized/test/img+img/"],
        # 'vqa': ['data/in/downstream_tasks/vqa/validation/q&a/']
        }
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bart_name", default="facebook/bart-large")
    parser.add_argument("--normalize_sentence_probs",
                        action="store_true", default=True)
    parser.add_argument("--uniform_sentence_probs",
                        action="store_true", default=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--reranked_output_file_name",
                        default="generations.rerank.txt")
    parser.add_argument("--beam_output_file_name",
                        default="generations.beam.txt")
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument(
        "--sentence_transformer_name", default="distilroberta-base-paraphrase-v1"
    )
    parser.add_argument(
        "--device", default="cuda"
    )
    parser.add_argument("--beam_size", default="4", type=int)

    parser.add_argument("--num_return_sequences", default="4", type=int)
    # required=True, choices=TEST_MAPPING.keys())
    parser.add_argument("--task", required=True)
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    (
        bart_name,
        output_file_name,
        beam_output_file_name,
        checkpoint_path,
        normalize_sentence_probs,
        uniform_sentence_probs,
        sentence_transformer_name,
        beam_size,
        num_return_sequences,
        device,
        task,
        config_path
    ) = (
        args.bart_name,
        args.reranked_output_file_name,
        args.beam_output_file_name,
        args.checkpoint_path,
        args.normalize_sentence_probs,
        args.uniform_sentence_probs,
        args.sentence_transformer_name,
        args.beam_size,
        args.num_return_sequences,
        args.device,
        args.task,
        args.config_path
    )
    test_paths = TEST_MAPPING[task]
    output_file_name = task + "." + output_file_name
    beam_output_file_name = task + "." + beam_output_file_name
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    varargs = dict(vars(args))
    varargs.pop("reranked_output_file_name")
    varargs.pop("beam_output_file_name")
    varargs.pop("checkpoint_path")
    varargs.pop("device")
    varargs.pop("bart_name")
    # varargs.pop("lxmert_name")
    varargs.pop("uniform_sentence_probs")
    varargs.pop("normalize_sentence_probs")
    output_file_name = output_file_name.replace(".txt", "") + "-" \
        + "-".join([k + "=" + str(v) for k, v in varargs.items()]) + ".txt"
    beam_output_file_name = beam_output_file_name.replace(".txt", "") + "-" \
        + "-".join([k + "=" + str(v).replace('/', '_') for k, v in varargs.items()]) + ".txt"

    beam_output_file = "/".join(checkpoint_path.split("/")[:-1]) + "/" \
        + beam_output_file_name

    training_params = get_training_params(config_path)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = MGMFullBart.from_pretrained(
        bart_name, **training_params)

    state_dict = checkpoint["state_dict"]
    state_dict = {re.sub("model.", "", k, 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    print('-'* 100)
    print('Script params:')
    print('-'* 100)
    print('\n'.join(f'{k}: {v}' for k, v in vars(args).items()))
    print('-'* 100)
    dataset = FiniteFolderDataset(test_paths[0], name=task, img_source = training_params.img_source, shuffle=False,
            encoder_pad_token_id=bart_tokenizer.pad_token_id, decoder_pad_token_id=bart_tokenizer.pad_token_id)
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.get_batch_fun(),
    )

    outputs = generate(
        model,
        data_loader,
        beam_size=beam_size,
        num_return_sequences=num_return_sequences,
        tokenizer=bart_tokenizer,
        normalize_sentence_probs=normalize_sentence_probs,
        uniform_sentence_probs=uniform_sentence_probs,
        device=device,
        max_length= args.max_length,
        min_length = args.min_length
    )
    # sentence_embedder = SentenceTransformer(sentence_transformer_name).cpu()

    lines = []
    with open(beam_output_file, "w") as beam_writer:
        for generations in tqdm(outputs, desc="Reranking"):
            first_beam_generation = generations["generations"][0]['output']
            print(first_beam_generation)
            beam_writer.write(
                generations["id"] + "\t" + first_beam_generation + "\n")
            """
            rescored_generations = mbrr_rescoring(
                sentence_embedder, generations["generations"]
            )
            generations["generations"] = rescored_generations
            min_scored = sorted(
                rescored_generations, key=lambda elem: -
                elem["sentence_probability"]
            )[0]
            min_rescored = rescored_generations[0]
            writer.write(generations["id"] + "\t" +
                         min_rescored["output"] + "\n")
            """
            lines.append({"image_id": generations["id"], "caption": first_beam_generation})
    
    with open(beam_output_file.replace(".txt", ".json"), "w") as writer:
        json.dump(lines, writer)
