import argparse
import numpy as np
from src.utils.bn_utils import get_bn2lemma
from tqdm import tqdm
import _pickle as pkl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bn2lemma_file", type=str,
                        default='/home/bianca/PycharmProjects/SensEmBERT/data/in/lemma2bnsynset.en.txt')
    parser.add_argument("--bn2gloss_file", type=str,
                        default='/home/tommaso/bn_offset_to_gloss.txt')
    parser.add_argument("--syns_file", type=str,
                        default='/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/imagenet/imagenet.bns.train.pkl')
    parser.add_argument("--out_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/in/imagenet.words.dm.new_gloss.txt')


    args = parser.parse_args()
    bn2lemma_file = args.bn2lemma_file
    bn2gloss_file = args.bn2gloss_file
    syns_file = args.syns_file
    out_file = args.out_file

    bn2lemma = get_bn2lemma(bn2lemma_file)
    bn2gloss = dict()
    with open(bn2gloss_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2gloss[fields[0]] = fields[1].strip() + '.'

    if syns_file.endswith('.npz'):
        synsets = set(np.load(syns_file).get('synsets'))
        tok_id = 'word+img.babelpic.{}'
    else:
        with open(syns_file, 'rb') as reader:
            synsets = pkl.load(reader)
            tok_id = 'word+img.imagenet.{}'

    counter = 0
    with open(out_file, 'wt') as writer:
        for img_bn in tqdm(synsets, desc="Creating dataset"):
            bn = img_bn.split('_')[0]
            for lemma in bn2lemma[bn]:
                writer.write('{}\t{}\t{}\t0-{}\t{}\t{}\t{}\n'.format(tok_id.format(counter), ' '.join(lemma.split('_')), lemma, len(lemma.split('_')), bn, bn2gloss[bn].capitalize()
                , bn))
                counter += 1
