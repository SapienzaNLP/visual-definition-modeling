import argparse
import subprocess

from tqdm import tqdm
import os
import _pickle as pkl
from src.utils.bn_utils import get_lemma2sensekey, read_bn2wn, read_wn2bn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import xml.etree.ElementTree as ET 

pos_dict = {'ADV': 'r', 'ADJ': 'a', 'NOUN': 'n', 'VERB': 'v'}

def compute_sentence_embedding(model, sense2gloss, batch_size=4):
    sent2embed = dict()
    for i in tqdm(range(0, len(sense2gloss), batch_size), desc="Generating sentencebert embeddings"):
        embeds = model.encode([sense2gloss[x] for x in sorted(sense2gloss)[i: i+batch_size]])
        for idx, e in zip(sorted(sense2gloss)[i: i+batch_size], embeds):
            sent2embed[idx] = e
    return sent2embed

def read_test_set(test_set_file):
    tree = ET.parse(test_set_file)
    root = tree.getroot()
    sentences = list()
    sentence_ids = list()
    sentence_labels = list()
    sentences_indices = list()
    inst2lemma = dict()
    for sentence in root.findall('./text/'):
        sent = list()
        labels = list()
        indices = list()
        sentence_ids.append(sentence.attrib['id'])
        index = 0
        for tok in sentence:
            if tok.text is None:
                continue
            if '_' in tok.text:
                tokenized_text = tok.text.split('_')
            else:
                tokenized_text = tok.text.split()
            if tok.tag == 'instance':
                instance_id = tok.attrib['id']
                pos = pos_dict[tok.attrib['pos']]
                inst2lemma[instance_id] = (tok.attrib['lemma'].lower(), pos)
                labels.append(instance_id)
                indices.append((index, index + len(tokenized_text)))
            index += len(tokenized_text)
            sent.extend(tokenized_text)
        sentences.append(sent)
        sentence_labels.append(labels)
        sentences_indices.append(indices)
    return sentences, sentence_ids, sentence_labels, sentences_indices, inst2lemma


def read_gold(gold_file):
    id2gold = dict()
    with open(gold_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split()
            id2gold[fields[0]] = set(fields[1:])
    return id2gold


def read_sense2gloss(input_file):
    sense2gloss = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            #sense2gloss[fields[0]] = [f.split('::')[0].lower().strip() for f in fields[2:] if f.split('::')[-1] == 'WN'][0]
            g = fields[1].capitalize()
            if not g.endswith("."):
                g = g + "."
            sense2gloss[fields[0]] = g

    return sense2gloss


def read_id2pred_gloss(input_file):
    id2pred_gloss = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            id2pred_gloss[fields[0]] = fields[1].strip()
    return id2pred_gloss


def write_prediction(out_file, instance2pred):
    with open(out_file, 'wt') as writer:
        for instance in instance2pred:
            writer.write('{} {}\n'.format(instance, instance2pred[instance][0]))


def predict_senses(inst2lemma, id2pred_gloss, idx2predembed, sense2gloss, sense2gloss_embed, lemma2target):
    inst2pred = dict()
    bn_glosses = list()
    generated_glosses = list()
    senses = list()
    instances = list()
    for inst in tqdm(inst2lemma, desc="Computing similarities"):
        instance_info = inst2lemma[inst]
        lemmapos = instance_info[0] + '#' + instance_info[1]
        if inst not in idx2predembed:
            continue
        gen_gloss_embed = idx2predembed[inst]
        sim = -1
        target_sense = None
        target_gloss = None
        for sense in sorted(lemma2target[lemmapos]):
            cos_sim = cosine_similarity([gen_gloss_embed], [sense2gloss_embed[sense]])[0][0]
            if cos_sim > sim:
                sim = cos_sim
                target_sense = sense
                target_gloss = sense2gloss[sense]
        inst2pred[inst] = (target_sense, sim, target_gloss, id2pred_gloss[inst])

    return inst2pred


def write_erros(out_file, inst2lemma, id2gold, inst2pred, sense2gloss, id2sent):

    with open(out_file, 'wt') as writer:
        for inst in sorted(id2gold, key=lambda x: inst2lemma[x][1]):
            if inst in inst2pred and inst2pred[inst][0] not in id2gold[inst]:
                if inst in inst2pred:
                    pred, score, target_gloss, gen_gloss = inst2pred[inst]
                    sent = id2sent['.'.join(inst.split('.')[:-1])]
                else:
                    pred = 'U'
                    score = 'U'
                    target_gloss = 'U'
                    gen_gloss = 'U'
                    sent = 'U'
                #if pred not in id2gold[inst]:
                writer.write('{}\t{}\t{}\n{}\n{}\n{}\n{}\n\n'.format(inst, pred, '\t'.join(id2gold[inst]), gen_gloss,
                                                                         target_gloss, '\t'.join([sense2gloss[i] for i in id2gold[inst]]),
                                                                         ' '.join(sent)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer_path", type=str,
                        default='data/in/WSD_Evaluation_Framework/Evaluation_Datasets/')
    #parser.add_argument("--test_set_file", type=str,
    #                    default='/home/bianca/Downloads/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.data.xml')
    #parser.add_argument("--gold_file", type=str,
    #                    default='/home/bianca/Downloads/WSD_Evaluation_Framework/Evaluation_Datasets/{}/{}.gold.key.txt')
    parser.add_argument("--out_dir", type=str,
                        default='data/out/wsd/')
    #parser.add_argument("--out_file_debug", type=str,
    #                    default='/home/bianca/PycharmProjects/MSECgeneration/data/out/wsd/{}_pred_debug_{}_gloss.txt')
    parser.add_argument("--generated_glosses_file", type=str, required=True)
                        # default='/home/bianca/wsd.generations.beam-sentence_transformer_name=distilroberta-base-paraphrase-v1-beam_size=4-num_return_sequences=4-task=wsd.txt')
    parser.add_argument("--glosses_file", type=str,
                        default='data/in/bn_offset_to_gloss.txt')
    parser.add_argument("--wn2bn_file", type=str,
                        default='data/in/all_bn_wn_keys.txt')
    parser.add_argument("--sbert_model", type=str,
                        default='distilroberta-base-paraphrase-v1')

    args = parser.parse_args()
    scorer_path = args.scorer_path
    test_set_file = os.path.join(args.scorer_path, "{}/{}.data.xml")# + args.test_set_file
    gold_file = test_set_file.replace("data.xml", "gold.key.txt") #args.gold_file
    generated_glosses_file = args.generated_glosses_file
    glosses_file = args.glosses_file
    out_file = os.path.join(args.out_dir, "{}_pred_{}_gloss.txt")
    out_file_debug = out_file.replace("_pred_", "_pred_debug_")#args.out_file_debug
    wn2bn_file = args.wn2bn_file
    sbert_model = args.sbert_model

    bn2wn = read_bn2wn(wn2bn_file)
    sense2gloss = read_sense2gloss(glosses_file)
    wn2bn = read_wn2bn(wn2bn_file)
    lemma2target = get_lemma2sensekey(wn2bn)

    datasets = ['ALL']
    model = SentenceTransformer(sbert_model)
    embed_path = 'data/in/sense2gloss.{}.{}.pkl'.format(sbert_model, glosses_file.split('/')[-1].split('.')[0])
    if not os.path.isfile(embed_path):
            sense2embed_bn = compute_sentence_embedding(model, sense2gloss)
            sense2embed = dict()
            for bn in sense2embed_bn:
                for wn in bn2wn[bn]:
                    sense2embed[wn] = sense2embed_bn[bn]
            pkl.dump(sense2embed, open(embed_path, 'wb'))
            del sense2embed_bn
    else:
        sense2embed = pkl.load(open(embed_path, 'rb'))

    sense2gloss_new = dict()
    for bn in sense2gloss:
        for wn in bn2wn[bn]:
            sense2gloss_new[wn] = sense2gloss[bn]

    sense2gloss = sense2gloss_new

    for dataset in datasets:
        id2pred_gloss = read_id2pred_gloss(generated_glosses_file.format(dataset))
        id2gold = read_gold(gold_file.format(dataset, dataset))
        sentences, sentence_ids, _, _, inst2lemma = read_test_set(test_set_file.format(dataset, dataset))
        
        idx2predembed = compute_sentence_embedding(model, id2pred_gloss)

        inst2pred = predict_senses(inst2lemma, id2pred_gloss, idx2predembed, sense2gloss, sense2embed, lemma2target)

        out_file_formatted = os.path.abspath(out_file.format(dataset, generated_glosses_file.split('/')[-2]))
        gf = os.path.abspath(gold_file.format(dataset, dataset))
        
        write_prediction(out_file_formatted, inst2pred)
        print('java Scorer ' + gold_file.format(dataset, dataset) + ' ' + out_file_formatted)
        result = subprocess.check_output(['cd ' + scorer_path + ' && java Scorer ' + gf + ' ' + out_file_formatted], shell=True)
        result = result.rstrip().split()

        precision = float(result[1][:-1])
        recall = float(result[3][:-1])
        f1 = float(result[5][:-1])

        print("Dataset:{}\t{}\t{}\t{}".format(dataset, precision, recall, f1))
        write_erros(out_file_debug.format(dataset, '.'.join(generated_glosses_file.split('/')[-1].split('.')[:-1])), inst2lemma, id2gold, inst2pred, sense2gloss, dict([(s_id, s) for s_id, s in zip(sentence_ids, sentences)]))
