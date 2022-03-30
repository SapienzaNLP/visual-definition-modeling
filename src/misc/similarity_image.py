import argparse
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

from src.utils.frcnn_utils import get_data
from tqdm import tqdm


def load_glosses(in_file):
    bn2glosses = dict()
    with open(in_file) as lines:
        for line in tqdm(lines, desc="loading glosses"):
            fields = line.rstrip().split("\t")
            bn2glosses[fields[0]] = [f.split('::')[0] for f in fields[1:]]
    return bn2glosses


def cos_sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_file", type=str,
                        default='/media/bianca/rocky/MultimodalGlosses/babelpic.frcnn.semcor.npz')
    parser.add_argument("--glosses_file", type=str,
                        default='/home/tommaso/bn.synset2glosses.en.txt')
    parser.add_argument("--obj_url", type=str,
                        default='https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt')
    parser.add_argument("--attr_url", type=str,
                        default='https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt')
    parser.add_argument('--out_file', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/out/babelpic.top100.similarities.txt")
    parser.add_argument('--batch_size', type=int,
                        default=30000)
    parser.add_argument('--num_target_img', type=int,
                        default=10000)
    parser.add_argument('--top_k', type=int,
                        default=100)

    args = parser.parse_args()
    image_file = args.image_file
    glosses_file = args.glosses_file
    obj_url = args.obj_url
    attr_url = args.attr_url
    out_file = args.out_file
    batch_size = args.batch_size
    num_target_img = args.num_target_img
    top_k = args.top_k

    bn2gloss = load_glosses(glosses_file)

    id2obj = dict([(i, v) for i, v in enumerate(get_data(obj_url))])
    id2attr = dict([(i, v) for i, v in enumerate(get_data(attr_url))])

    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    images = np.load(image_file)
    flat_images = list()
    flat_syns = list()
    flat_attr = list()
    flat_obj = list()
    for imgs, syn, attrs, objs in zip(images['features'], images['synsets'], images['attrs'], images['objs']):
        flat_images.extend(imgs)
        flat_syns.extend([syn for i in range(len(imgs))])
        flat_attr.extend([id2attr[a] for a in attrs[0]])
        flat_obj.extend([id2obj[o] for o in objs[0]])

    syn2sim = OrderedDict()
    target_imgs = torch.Tensor(flat_images[:num_target_img]).cuda()

    step = 0
    for k in tqdm(range(0, len(flat_images), batch_size)):
        oth_images = torch.Tensor(flat_images[k: k+batch_size]).cuda()
        sims = cos_sim_matrix(target_imgs, oth_images).detach().cpu().numpy().tolist()
        for i, sim in enumerate(sims):
            syn_sim = syn2sim.get(i, list())
            syn_sim.extend(sim)
            syn2sim[i] = syn_sim
        del oth_images
        step += 1
        if step == 4:
            break

    with open(out_file, 'wt') as writer:
        for i in syn2sim:
            writer.write('{}\t{}\t{}\t{}\n{}\n'.format(flat_syns[i], flat_obj[i], flat_attr[i],
                                                       bn2gloss[flat_syns[i].split('_')[0]][0], '-'*10))
            sims = syn2sim[i]
            most_sims, most_idx = torch.topk(torch.Tensor(sims), top_k)
            for k, s in zip(most_idx, most_sims):
                writer.write('{}\t{}\t{}\t{}\t{}\n'.format(s, flat_syns[k], flat_obj[k], flat_attr[k],
                                                           bn2gloss[flat_syns[k].split('_')[0]][0]))
            writer.write('\n\n')




