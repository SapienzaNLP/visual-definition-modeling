import argparse
import spacy
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

def convert_reverse_pos(pos):
    pos_dict = {'NOUN': '1', 'PROPN': '1', 'VERB' :'2', 'ADJ': '3', 'ADV': '4'}
    return pos_dict[pos] if pos in pos_dict else '6'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_file", type=str,
                        default='/home/bianca/PycharmProjects/MSECgeneration/data/in/dm/Examples/examples_CHANG[train_easy].txt')
    parser.add_argument("--definition_file", type=str,
                        default='/home/bianca/PycharmProjects/MSECgeneration/data/in/dm/Definitions/definitions_CHANG.txt')
    parser.add_argument("--out_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/in/dm/chang.easy.train.dm.txt')

    args = parser.parse_args()
    example_file = args.example_file
    definition_file = args.definition_file
    out_file = args.out_file

    id2def = dict()
    with open(definition_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            id2def[fields[0]] = fields[1]

    id2info = dict()
    counter = 0
    with open(example_file, 'rt') as lines:
        for line in tqdm(lines, desc="Postagging dataset"):
            fields = line.rstrip().split('\t')
            definition_id = fields[0]
            target_start_index = -1
            target_end_index = -1
            target_word = None
            sent = list()
            for i, word in enumerate(fields[1].split()):
                if word.startswith('@#*'):
                    w = word[3:].split('|')[0]
                    sent.append(w)
                    target_start_index = i
                    target_end_index = i + len(w.split())
                    target_word = w
                else:
                    sent.append(word)
            tok_id = 'txt.dm.{}.{}'.format(counter, target_word)
            id2info[tok_id] = (' '.join(sent), target_word, '-'.join([str(target_start_index), str(target_end_index)]), definition_id, id2def[definition_id], definition_id)
            counter += 1

    with open(out_file, 'wt') as writer:
        for i in id2info:
            writer.write('{}\t{}\n'.format(i, '\t'.join(id2info[i])))