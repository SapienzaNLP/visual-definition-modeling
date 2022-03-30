import torch
from src.modeling.vl_model_wrapper import *
from transformers import AutoTokenizer
from src.data_preparation.multimodal_datasets_lazy import (
    CircularSampler,
    CircularSamplerWithCoherentAccumulation,
    CurriculumSampler,
    DatasetAlternator,
    FolderDataset,
    ModalitySampler, PicklesDataset,
)
from torch.utils.data.dataloader import DataLoader
import torch
if __name__ == '__main__':
    bert_model = 'bert-base-uncased'
    config_file = 'config/vl-bert-base.json'
    config = BertConfig.from_json_file(config_file)
    type_vocab_size = config.type_vocab_size
    config.type_vocab_size = 2
    checkpoint_path = '/home/bianca/PycharmProjects/MultimodalGlosses/data/out/pretraining/wandb/run-20210427_173912-1nsa2vj1/files/checkpoints/global_step=49999.ckpt'
    model = BertForVLGeneration.from_pretrained(checkpoint_path, config=config,
                                                    from_hf=True)
    config.type_vocab_size = type_vocab_size
    
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    input_ids = [tokenizer.encode('word + img: <define> basket ash </define>')]
    input_ids = torch.LongTensor(input_ids * 10)
    visual_feats = torch.rand(10, 36, 2048)
    visual_pos = torch.rand(10, 36, 4)
    decoder_input_ids = [tokenizer.encode('and this is the definition')]
    decoder_input_ids = torch.LongTensor(decoder_input_ids * 10)

    out = model(input_ids=input_ids, visual_feats=visual_feats, visual_pos=visual_pos, visual_attention_mask=torch.ones(len(visual_feats), len(visual_feats[0])), 
    decoder_input_ids=decoder_input_ids, labels=decoder_input_ids, attention_mask=torch.ones(len(input_ids), len(input_ids[0])))

    generation_params = dict(
            num_beams=4,
            min_length=5,
            max_length=15,
            temperature=1,
            repetition_penalty=2,
            length_penalty=1.5,
            no_repeat_ngram_size=2,
        )
    out = model.generate(
                    input_ids=input_ids,
                    **generation_params,
                    decoder_start_token_id=tokenizer.cls_token_id,
                    decoder_kwargs={},
                    encoder_kwargs={'input_imgs':visual_feats, 'image_loc':visual_pos}
                )
    print(tokenizer.decode(out[0]))