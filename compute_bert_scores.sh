#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=16000M
# we run on the gpu partition
#SBATCH -p gpu --gres=gpu:titanx:1
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10:00:00
#echo "vl-bert-pretrained-bart word+img research/multimodal_glosses/32lede2c"
#./src/evaluation/bert-score-rescaled.sh data/in/word+img.test.new.dm.txt data/out/vl-bert-bart-pretrained/word+img/wandb/run-20210514_232256-32lede2c/files/checkpoints/word+img.generations.beam-max_length\=64-min_length\=1-beam_size\=4-num_return_sequences\=4-task\=word+img-base_model\=vl-bert.txt  en

echo "bart-lxmert word+img research/multimodal_glosses/2llkimow"
./src/evaluation/bert-score-rescaled.sh \
 data/in/word+img.test.new.dm.txt \
 ~/computerome/data/out/pretraining-plain-no-task-warmup-bart-lxmert/word+img/wandb/run-20210513_190220-2llkimow/files/checkpoints/word+img.generations.beam-max_length\=512-min_length\=1-sentence_transformer_name\=distilroberta-base-paraphrase-v1-beam_size\=4-num_return_sequences\=4-task\=word+img-config_path\=config_word+img_config.bart-lxmert.json.txt \
 en

echo "bart-frcnn word+img research/multimodal_glosses/1so9e58v"
./src/evaluation/bert-score-rescaled.sh \
 data/in/word+img.test.new.dm.txt \
 data/out/pretraining-plain-no-task-warmup-bart-frcnn/wandb/run-20210509_175606-1so9e58v/files/checkpoints/word+img.generations.beam-max_length\=512-min_length\=1-sentence_transformer_name\=distilroberta-base-paraphrase-v1-beam_size\=4-num_return_sequences\=4-task\=word+img-config_path\=config_word+img_config.bart-frcnn.json.txt \
 en

echo "vl-bert img+img research/multimodal_glosses/1so9e58v"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 ~/computerome/data/out/img+img/vlbert/wandb/run-20210512_203337-3gx012en/files/checkpoints/img+img.generations.beam-max_length\=64-min_length\=1-beam_size\=4-num_return_sequences\=4-task\=img+img-base_model\=vl-bert.txt \
 en

echo "lxmert img+img research/multimodal_glosses/13d4oys9"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 /home/npf290/computerome/data/out/img+img/lxmert/wandb/run-20210512_203524-13d4oys9/files/checkpoints/img+img.generations.beam-max_length=64-min_length=1-beam_size=4-num_return_sequences=4-task=img+img-base_model=lxmert.txt \ 
 en

echo "vl-bert-pretrained-bart img+img research/multimodal_glosses/27iudmn0"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 /home/npf290/computerome/data/out/vl-bert-bart-pretrained/img+img/wandb/run-20210513_154026-27iudmn0/files/checkpoints/img+img.generations.beam-max_length=64-min_length=1-beam_size=4-num_return_sequences=4-task=img+img-base_model=vl-bert.txt \
 en


echo "lxert-pretrained-bart img+img research/multimodal_glosses/1gatpfe9"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 /home/npf290/computerome/data/out/lxmert-bart-pretrained/img+img/wandb/run-20210514_205350-1gatpfe9/files/checkpoints/img+img.generations.beam-max_length=64-min_length=1-beam_size=4-num_return_sequences=4-task=img+img-base_model=lxmert.txt \
 en
echo "bart-vl-bert img+img research/multimodal_glosses/j98rqarg"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 /home/npf290/computerome/data/out/pretraining-plain-no-task-warmup-bart-lxmert/img+img/wandb/run-20210512_210300-j98rqarg/files/checkpoints/img+img.generations.beam-max_length=512-min_length=1-sentence_transformer_name=distilroberta-base-paraphrase-v1-beam_size=4-num_return_sequences=4-task=img+img-config_path=config_img+img_config.bart-lxmert.json.txt \
 en
echo "bart-lxmert img+img research/multimodal_glosses/2pc8acgl"
./src/evaluation/bert-score-rescaled.sh \
 data/in/mscoco.object.cap.karpathy.test.man_map.verified.dm.txt \
 /home/npf290/computerome/data/out/pretraining-plain-no-task-warmup-bart-frcnn/img+img/wandb/run-20210513_152627-2pc8acgl/files/checkpoints/img+img.generations.beam-max_length=512-min_length=1-sentence_transformer_name=distilroberta-base-paraphrase-v1-beam_size=4-num_return_sequences=4-task=img+img-config_path=config_img+img_config.bart-frcnn.json.txt \
 en
