import argparse

from src.utils.modeling_frcnn import GeneralizedRCNN
from src.utils.frcnn_utils import Config, get_data
from src.utils.processing_image import Preprocess
import os
from tqdm import tqdm
import _pickle as pkl
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str,
                        default='/media/bianca/fluffy_potato/mscoco/images/val2017/')
    parser.add_argument("--obj_url", type=str,
                        default='https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt')
    parser.add_argument("--attr_url", type=str,
                        default='https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt')
    parser.add_argument('--out_file', type=str,
                        default="/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.frcnn.boxes.{}.npz")
    parser.add_argument('--target_bns_file', type=str,
                        default="/media/bianca/fluffy_potato/MultimodalGlosses/all.bns.txt")
    parser.add_argument('--wn2bn_file', type=str,
                        default="/home/tommaso/WSDFramework20/resources/mappings/all_bn_wn.txt")
    parser.add_argument("--batch_size", type=int,
                        default=1)
    parser.add_argument("--batch_size_save", type=int,
                        default=50000)
    parser.add_argument("--babelpic", type=bool,
                        default=True)

    args = parser.parse_args()
    image_dir = args.image_dir
    obj_url = args.obj_url
    attr_url = args.attr_url
    out_file = args.out_file
    target_bns_file = args.target_bns_file
    wn2bn_file = args.wn2bn_file
    batch_size = args.batch_size
    batch_size_save = args.batch_size_save
    babelpic = args.babelpic

    if target_bns_file.endswith('.pkl'):
        with open(target_bns_file, 'rb') as reader:
            target_bns = pkl.load(reader)
    else:
        target_bns = set()
        with open(target_bns_file, 'rt') as lines:
            for line in lines:
                target_bns.add(line.rstrip())
    
    wn2bn = dict()
    with open(wn2bn_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            wn2bn[fields[1][-1] + fields[1][:-1]] = fields[0]

    computed_imgs = set()
    #computed_imgs = np.load('/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/MultimodalGlosses/babelpic.frcnn.all_imgs.npz')
    #computed_imgs = set(computed_imgs.get('synsets'))

    objids = get_data(obj_url)
    attrids = get_data(attr_url)

    frcnn_cfg = Config.from_pretrained('unc-nlp/frcnn-vg-finetuned')
    frcnn = GeneralizedRCNN.from_pretrained('unc-nlp/frcnn-vg-finetuned', config=frcnn_cfg).to('cuda')
    image_preprocess = Preprocess(frcnn_cfg)

    if babelpic:
        image_files_all = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                 if os.path.isfile(os.path.join(image_dir, f)) and (f.endswith('.jpg')
                                                                                    or f.endswith('.png')
                                                                                    or f.endswith('.JPG')
                                                                                    or f.endswith('.JPEG'))
                                                                                    and f not in computed_imgs])
                          #and f.split('_')[0] in target_bns])
    else:
        image_files_all = sorted([os.path.join(image_dir, d, f) for d in os.listdir(image_dir) for f in os.listdir(os.path.join(image_dir, d)) 
                                                        if os.path.isfile(os.path.join(image_dir, d, f)) and (f.endswith('.jpg')
                                                                                    or f.endswith('.png')
                                                                                    or f.endswith('.JPG')
                                                                                    or f.endswith('.JPEG'))
                                                                                    and f not in computed_imgs])
    covered_bns = dict()
    all_syns = set()
    image_files = list()
    for f in image_files_all:
        if babelpic:
            syn =f.split('/')[-1].split('_')[0]
        else: 
            syn = wn2bn[f.split('/')[-1].split('_')[0]]
        all_syns.add(syn)
        if syn not in covered_bns or covered_bns[syn] < 5:
            image_files.append(f)
            counter = covered_bns.get(syn, 0)
            counter += 1
            covered_bns[syn] = counter


    print("All synsets: {}\nImages to encode: {}\nCovered bns: {}".format(len(all_syns),
                                                                          len(image_files), len(covered_bns)))
    objs = list()
    attrs = list()
    normalized_boxes = list()
    features = list()
    synsets = list()
    boxes = list()
    file_id = 3
    for image_batch in tqdm(range(0, len(image_files), batch_size), desc="Computing image embeddings"):

        images, sizes, scales_yx = image_preprocess(image_files[image_batch: image_batch + batch_size])

        try:
            output_dict = frcnn(images.to('cuda:0'), sizes, scales_yx=scales_yx, padding="max_detections",
                            max_detections=frcnn_cfg.max_detections,
                            return_tensors="pt")
        except:
            continue

        objs.extend([(o, p) for o, p in zip(output_dict.get('obj_ids').detach().cpu().numpy(),
                                            output_dict.get('obj_probs').detach().cpu().numpy())])
        attrs.extend([(a, p) for a, p in zip(output_dict.get('attr_ids').detach().cpu().numpy(),
                                             output_dict.get('attr_probs').detach().cpu().numpy())])
        normalized_boxes.extend(output_dict.get('normalized_boxes').detach().cpu().numpy())
        features.extend(output_dict.get('roi_features').detach().cpu().numpy())
        synsets.extend([wn2bn[x.split('/')[-1].split('_')[0]] + '_' + x.split('/')[-1] for x in image_files[image_batch: image_batch + batch_size]])
        synsets.extend([x.split('/')[-1] for x in image_files[image_batch: image_batch + batch_size]])
        boxes.extend(output_dict.get('boxes').detach().cpu().numpy())
        if (len(synsets) % batch_size_save) == 0:
            np.savez_compressed(out_file.format(file_id), synsets=synsets, objs=objs, attrs=attrs, normalized_boxes=normalized_boxes,
                        features=features)
            file_id += 1
            objs = list()
            attrs = list()
            normalized_boxes = list()
            features = list()
            synsets = list()
            boxes = list()

    if (len(synsets)) >= 1:
            np.savez_compressed(out_file.format(file_id), synsets=synsets, objs=objs, attrs=attrs, normalized_boxes=normalized_boxes,
                        features=features)

"""
                image_files[image_batch: image_batch + batch_size],
                output_dict.get("boxes"),
                output_dict.pop("obj_ids"),
                output_dict.pop("obj_probs"),
                output_dict.pop("attr_ids"),
                output_dict.pop("attr_probs"))
"""