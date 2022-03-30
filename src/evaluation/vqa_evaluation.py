from argparse import ArgumentParser
from typing import Dict
from src.VQA.PythonHelperTools.vqaTools.vqa import VQA
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os
from src.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
from pprint import pprint



def evaluate(prediction_file:Dict, annoations_path:str, questions_path:str):
    vqa = VQA(annoations_path, questions_path)
    vqaRes = vqa.loadRes(prediction_file, questions_path)
    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 
    # save evaluation results to ./Results folder
    # json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    # json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    # json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    # json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))
    return vqaEval

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--predictions', default='/media/ssd/tommaso/data/out/vqa_2/wandb/run-20210218_162914-3ddhlnfl/files/checkpoints/vqa.generations.beam-max_length=3-min_length=1-sentence_transformer_name=distilroberta-base-paraphrase-v1-beam_size=1-num_return_sequences=1-task=vqa.json')
    args = parser.parse_args()
    prediction_file = args.predictions #'src/VQA/Results/OpenEnded_mscoco_train2014_fake_results.json'
    anotations_path = 'data/in/downstream_tasks/vqa/annotations/v2_mscoco_val2014_annotations.json'
    questions_path = 'data/in/downstream_tasks/vqa/questions/v2_OpenEnded_mscoco_val2014_questions.json'
    eval = evaluate(prediction_file, annoations_path=anotations_path, questions_path=questions_path)
    pprint(eval.accuracy)
