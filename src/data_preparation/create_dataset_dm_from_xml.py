import argparse
from xml.etree import ElementTree as ET

from src.utils.bn_utils import read_wn2bn
import _pickle as pkl

def load_synset2gloss(input_file, lang, sources):
    bn2gloss = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn = fields[0]
            glosses = list()
            for gloss in fields[1:]:
                gloss = gloss.lower()
                if gloss.split('::')[-1] not in sources:
                    continue
                if gloss.split('::')[-2] == lang:
                    trimmed_gloss = '::'.join(gloss.split('::')[:-2])
                    if len(trimmed_gloss.strip()) == 0:
                        continue
                    glosses.append(trimmed_gloss)
            bn2gloss[bn] = glosses
    return bn2gloss

def read_id2gold(input_file):
    id2gold = dict()
    with open(input_file) as lines:
        for line in lines:
            fields = line.rstrip().split()
            id2gold[fields[0]] = fields[1]
    return id2gold


def semcor_parser(input_file, id2gold, wn2bn):
    tree = ET.parse(input_file)
    root = tree.getroot()
    documents = root.findall("./text")
    tok_ids = list()
    sentences = list()
    lemmas = list()
    senses = list()
    indices = list()
    for doc in documents:
        for xml_sentence in doc.findall('./sentence'):
            tok_ids_sentence = list()
            lemmas_sentence = list()
            senses_sentence = list()
            indices_sentence = list()
            sentence = list()
            curr_ind = 0
            for tok in xml_sentence:
                if tok.text is None:
                    continue
                text = tok.text.replace('_', ' ').split()
                sentence.extend(text)
                if "id" in tok.attrib:
                    tok_ids_sentence.append(tok.attrib['id'])
                    lemmas_sentence.append(tok.attrib['lemma'])
                    indices_sentence.append((curr_ind, curr_ind + len(text)))
                    senses_sentence.append(wn2bn[id2gold[tok.attrib['id']]])
                curr_ind += len(text)
            tok_ids.append(tok_ids_sentence)
            lemmas.append(lemmas_sentence)
            sentences.append(sentence)
            senses.append(senses_sentence)
            indices.append(indices_sentence)

    return tok_ids, lemmas, sentences, senses, indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--semcor_gold_file", type=str,
                        default='/home/bianca/Downloads/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.gold.key.txt')
    parser.add_argument("--semcor_data_file", type=str,
                        default='/home/bianca/Downloads/WSD_Evaluation_Framework/Evaluation_Datasets/senseval2/senseval2.data.xml')
    parser.add_argument("--gloss_file", type=str,
                        default='/home/tommaso/bn_offset_to_gloss.txt')
    parser.add_argument("--wn2bn_file", type=str,
                        default='/media/bianca/storage1/backup/wikicat/data/in/all_bn_wn_keys.txt')
    parser.add_argument("--out_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/in/senseval2.data.dm.new_gloss.txt')


    args = parser.parse_args()
    semcor_gold_file = args.semcor_gold_file
    semcor_data_file = args.semcor_data_file
    gloss_file = args.gloss_file
    wn2bn_file = args.wn2bn_file
    out_file = args.out_file

    wn2bn = read_wn2bn(wn2bn_file)

    bn2gloss = dict()
    with open(gloss_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2gloss[fields[0]] = fields[1]

    id2gold = read_id2gold(semcor_gold_file)
    tok_ids, lemmas, sentences, senses, indices = semcor_parser(semcor_data_file, id2gold, wn2bn)

    with open(out_file, 'wt') as writer:
        for sentence, tok_id, lemma, sense, index in zip(sentences, tok_ids, lemmas, senses, indices):
            for t, l,  s, i in zip(tok_id, lemma, sense, index):
                writer.write('{}\t{}\t{}\t{}-{}\t{}\t{}\t{}\n'.format(t, ' '.join(sentence), l, i[0], i[1], s, bn2gloss[s], ','.join(sense)))

