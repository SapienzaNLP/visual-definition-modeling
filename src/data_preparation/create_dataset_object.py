import argparse
from src.utils.modeling_frcnn import norm_box
from src.utils.frcnn_utils import Config
from src.utils.processing_image import Preprocess
import numpy as np
from tqdm import tqdm
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_file", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.frcnn.boxes.{}.npz')
    parser.add_argument("--object_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_object.all.txt')
    parser.add_argument("--image_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_imgid.all.txt')
    parser.add_argument("--bn2wn_file", type=str,
                        default='/home/tommaso/all_bn_wn.txt')
    parser.add_argument("--gloss_file", type=str,
                        default='/home/tommaso/bn_offset_to_gloss.txt')
    parser.add_argument("--ann_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/ewiser_predictions/mscoco.caption.object.all.results.key.txt')
    parser.add_argument("--window_size", type=int,
                        default=50)
    parser.add_argument("--out_file", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.object.cap.all.dm.txt')
    parser.add_argument("--cap", type=int,
                        default=5000)

    args = parser.parse_args()
    embedding_file = args.embedding_file
    object_file = args.object_file
    image_file = args.image_file
    bn2wn_file = args.bn2wn_file
    gloss_file = args.gloss_file
    ann_file = args.ann_file
    window_size = args.window_size
    out_file = args.out_file
    cap = args.cap

    id2file = dict()
    with open(image_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            id2file[fields[0]] = fields[1]

    id2objs = dict()
    with open(object_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            imgid = id2file[fields[0]]
            objs = id2objs.get(imgid, list())
            objs.append((fields[1], fields[2], [float(x) for x in fields[4:]]))
            id2objs[imgid] = objs

    wn2bn = dict()
    with open(bn2wn_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            wn2bn['wn:'+fields[1]] = fields[0]

    bn2gloss = dict()
    with open(gloss_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2gloss[fields[0]] = fields[1]

    id2ann = dict()
    with open(ann_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split()
            if len(fields) == 1:
                continue
            image_id = id2file[fields[0].split('.')[1][1:]]
            obj_id = fields[0].split('.')[2]
            objs = id2ann.get(image_id, dict())
            objs[obj_id] = bn2gloss[wn2bn[fields[1]]]
            id2ann[image_id] = objs

    ann2ids = dict()
    for i in id2ann:
        for o in id2ann[i]:
            anns = ann2ids.get(id2ann[i][o], list())
            anns.append((i, o))
            ann2ids[id2ann[i][o]] = anns
    
    for ann in ann2ids:
        if len(ann2ids[ann]) > cap:
            to_remove = np.random.choice(np.arange(len(ann2ids[ann])), len(ann2ids[ann]) - cap, replace=False)
            for idx in to_remove:
                i, o = ann2ids[ann][idx]
                id2ann[i].pop(o)
                if len(id2ann[i]) == 0:
                    id2ann.pop(i)

    id2index = dict()
    not_found = 0
    covered = 0
    all = 0
    file_id = 0
    for z in range(4):
        in_data = {}
        data = np.load(embedding_file.format(z))
        for image_id, boxes in tqdm(zip(data.get('synsets'), data.get('boxes')), desc="Reading boxes"):
            if image_id not in id2objs:
                not_found += 1
                continue
            for i, _, obj in id2objs[image_id]:
                all += 1
                x_left = obj[0]
                x_right = obj[0] + obj[2]
                y_down = obj[1]
                y_up = obj[1] + obj[3]
                cov = False
                for k, b in enumerate(boxes):
                    if abs(x_left - b[0]) <= window_size and abs(x_right - b[2]) <= window_size and abs(y_down - b[1]) <= window_size and abs(y_up - b[3]) <= window_size:
                        cov = True
                        id2index[(image_id, i)] = k
                if cov:
                    covered += 1
    
    with open(out_file, 'wt') as writer:
        for i in id2index:
            if i[0] not in id2ann or i[1] not in id2ann[i[0]]:
                continue
            writer.write('img+img.{}.{}\t<s>\t<s>\t{}-{}\t{}\t{}\t{}\n'.format(i[0].split('.')[0], i[1], str(id2index[i]), str(id2index[i]+1),
            i[0], id2ann[i[0]][i[1]], i[0]))