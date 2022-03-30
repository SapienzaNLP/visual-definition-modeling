import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_file", type=str,
                        default='/media/bianca/fluffy_potato/mscoco/dataset_coco.json')
    parser.add_argument("--dm_file", type=str,
                        default='/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.object.cap.all.dm.txt')
    parser.add_argument('--out_file', type=str,
                        default="/media/bianca/fluffy_potato/MultimodalGlosses/mscoco.object.cap.karpathy.{}.dm.txt")

    args = parser.parse_args()
    splits_file = args.splits_file
    dm_file = args.dm_file
    out_file = args.out_file

    splits_data = json.load(open(splits_file, 'rb'))

    splits = dict() 
    for i in splits_data['images']: 
        images = splits.get(i['split'], set()) 
        images.add(i['filename'].split('_')[-1]) 
        splits[i['split']] = images 

    id2img = dict()
    with open(dm_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            imgs = id2img.get(fields[4], list())
            imgs.append(line)
            id2img[fields[4]] = imgs

    for i in splits:
        if i != 'restval':
            with open(out_file.format(i), 'wt') as writer:
                for imgid in splits[i]:
                    if imgid not in id2img:
                        continue
                    for line in id2img[imgid]:
                        writer.write(line)
