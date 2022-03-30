from argparse import ArgumentParser
from src.modeling.components import prepare_encoder_input

from tqdm.std import tqdm
from src.commons.commons import move_batch
from src.training.model_full_bart_trainer import get_training_params
from src.modeling.model_full_bart_visual_feat import MGMFullBart
from src.data_preparation.multimodal_datasets_lazy import FiniteFolderDataset

from torch.utils.data.dataloader import DataLoader
from src.generation.generate import generate, score_generations
from transformers import BartTokenizerFast
import torch
import re
import json

def select_answers(model, data_loader, possible_answers):
    for batch in tqdm(data_loader):
        batch = move_batch(batch, device)
        ids = batch["ids"]
        answer_losses = list()
        for a in possible_answers:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                batch_kinds=batch["batch_kinds"],
                visual_feats=batch["visual_feats"],
                visual_attention_mask=batch["visual_attention_mask"],
                activate_img_features=batch["activate_img_features"],
                insert_img_objects=batch["insert_img_objects"],
                target_indexes=batch["target_indexes"] * 1,
                labels=a,
                return_dict=True,
                use_cache=False,
            )
            loss = out["loss"]
            answer_losses.append(loss.item())
        sorted_scored_answers = sorted(
            zip(ids, possible_answers, answer_losses), key=lambda elem: elem[1]
        )
        best_answer = sorted_scored_answers[0]
        return best_answer, sorted_scored_answers


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bart_name", default="facebook/bart-large")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--min_length", default=1, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam_size", default="1", type=int)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--annotation_file', required=True)
    
    args = parser.parse_args()
    (bart_name, checkpoint_path, beam_size, min_length, max_length, device,) = (
        args.bart_name,
        args.checkpoint_path,
        args.beam_size,
        args.min_length,
        args.max_length,
        args.device,
    )
    test_path = "data/in/downstream_tasks/vqa/validation/q&a/"
    bart_tokenizer = BartTokenizerFast.from_pretrained(bart_name)
    varargs = dict(vars(args))
    varargs.pop("checkpoint_path")
    varargs.pop("device")
    varargs.pop("bart_name")

    training_params = get_training_params()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = MGMFullBart.from_pretrained(bart_name, **training_params)

    state_dict = checkpoint["state_dict"]
    state_dict = {re.sub("model.", "", k, 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    print("-" * 100)
    print("Script params:")
    print("-" * 100)
    print("\n".join(f"{k}: {v}" for k, v in vars(args).items()))
    print("-" * 100)
    dataset = FiniteFolderDataset(
        test_path, name="q&a", shuffle=False, pad_token_id=bart_tokenizer.pad_token_id
    )

    data_loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        collate_fn=dataset.get_batch_fun(),
    )
    generation_params = dict(
        num_beams=beam_size,
        min_length=min_length,
        max_length=max_length,
        temperature=1,
        repetition_penalty=2,
        length_penalty=1,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    lines = []
    multichoice_answers = list()
    with open(args.annotation_file) as reader:
        annotations = json.load(reader)['annotations']
        for annotation in annotations:
            ans = annotation['multiple_choice_answer']
            multichoice_answers.append(ans)

    
    best_answer, scored_answers = select_answers(
        model, data_loader, multichoice_answers
    )
    lines.append(
        {"question_id": int(best_answer[0].split(".")[1]), "answer": best_answer[1]}
    )
    with open(args.output_path, "w") as writer:
            json.dump(lines, writer)
