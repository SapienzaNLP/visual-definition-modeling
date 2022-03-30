import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import LxmertTokenizer, BartTokenizer
import torch
from src.data_preparation.multimodal_datasets import MultimodalTxtDataset
from src.data_preparation.multimodal_sampler import IndexSampler, SamplerAlternator
from src.modeling.model import Lxmert_LM
from src.training.model_trainer import IMGDropoutLogger, get_training_params, LxmertLMFinetuner

if __name__ == "__main__":
    encoder_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    decoder_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    train_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                         decoder_tokenizer,
                                         "<d>", "</d>", "data/in/semcor.data.dm.all.txt",
                                         "data/in/babelpic.frcnn.semcor.all.npz",
                                         limit_sentences=1)
    dev_dataset = MultimodalTxtDataset(encoder_tokenizer,
                                       decoder_tokenizer,
                                       "<d>", "</d>", "data/in/semeval2007.data.dm.txt",
                                       "data/in/babelpic.frcnn.semcor.all.npz")
    lxmert_lm = Lxmert_LM()
    training_params = get_training_params(train_dataset)
    wandb_logger = WandbLogger("multimodal_gloss_modelling",
                               project="multimodal_glosses",
                               offline=True,
                               log_model=True,
                               save_dir="data_ssd4/out/")
    lr_logger = LearningRateMonitor()
    dropout_logger = IMGDropoutLogger()
    checkpoint_dir = os.path.join(wandb_logger.experiment.dir, "checkpoints/")
    os.makedirs(checkpoint_dir)
    pprint(training_params)
    checkpointer = ModelCheckpoint(checkpoint_dir, monitor="val_loss", save_top_k=-1, period=2, save_last=True)
    trainer = pl.Trainer(gpus=1, precision=16, max_steps=training_params.num_training_steps,
                         checkpoint_callback=checkpointer,
                         accumulate_grad_batches=training_params.gradient_accumulation,
                         logger=[wandb_logger], callbacks=[lr_logger, dropout_logger],
                         gradient_clip_val=training_params.gradient_clip_val,
                         num_sanity_val_steps=0
                         )  # , limit_val_batches=0.1)

    txt_img_indices = dev_dataset.txt_img_indices
    txt_only_indices = dev_dataset.txt_only_indices
    txt_img_indices_sampler = IndexSampler(txt_img_indices)
    txt_only_indices_sampler = IndexSampler(txt_only_indices)

    alternator = SamplerAlternator([txt_img_indices_sampler, txt_only_indices_sampler], 1, False)
    data_loader = DataLoader(dev_dataset, batch_sampler=alternator, num_workers=0,
                             collate_fn=dev_dataset.get_batch_fun())
    ckpt_path = 'data_ssd4/out/epoch=27.ckpt'
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cuda"))
    state_dict = checkpoint["state_dict"]
    for k, v in list(state_dict.items()):
        del state_dict[k]
        state_dict[k.replace("lxmertlm.", "")] = v
    lxmert_lm.load_state_dict(state_dict)
    finetuner = LxmertLMFinetuner(lxmert_lm, train_dataset, dev_dataset, training_params,
                                  encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer)

    print("testing")
    x = trainer.test(model=finetuner,
                 test_dataloaders=[data_loader])

    # print(x)
    lxmert_lm.cuda()
    print("generating")
    generation_params = dict(num_beams=4,
                             min_length=5,
                             max_length=15,
                             temperature=1,
                             repetition_penalty=2,
                             length_penalty=1.5,
                             no_repeat_ngram_size=2)
    lxmert_lm.eval()
    for batch in data_loader:
        beam_output = lxmert_lm.generate(
            batch["input_ids"].cuda(),
            **generation_params,
            decoder_start_token_id=decoder_tokenizer.bos_token_id,
            decoder_kwargs={},
            encoder_kwargs={"visual_feats": batch["visual_feats"].cuda(), "visual_pos": batch["visual_pos"].cuda(),
                            "visual_attention_mask": batch["visual_attention_mask"].cuda()}

        )
        print("Output:\n" + 100 * '-')
        print(encoder_tokenizer.decode(batch["input_ids"][0]))
        print(decoder_tokenizer.decode(beam_output[0]))
        print(decoder_tokenizer.decode(batch["labels"][0]).strip())

