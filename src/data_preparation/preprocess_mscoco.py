import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscoco_ann_dir", type=str,
                        default='/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/mscoco/annotations/captions_{}2017.json')
    parser.add_argument('--out_file_img', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_imgid.{}.txt")
    parser.add_argument('--out_file_caption', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_caption.{}.txt")

    args = parser.parse_args()
    mscoco_ann_dir = args.mscoco_ann_dir
    out_file_img = args.out_file_img
    out_file_caption = args.out_file_caption

    for i in ['train', 'val']:
        id2fname = dict()
        id2caption = dict()
        json_data = json.load(open(mscoco_ann_dir.format(i), 'r'))
        for j in json_data['images']:
            id2fname[j['id']] = j['file_name']
        for j in json_data['annotations']:
            if len(j['caption']) > 1:
                caption = j['caption'].strip()
                trans = str.maketrans('\n', ' ')
                caption = caption.translate(trans)
                id2caption[j['image_id']] = caption
        with open(out_file_img.format(i), 'wt') as writer:
            for id in id2fname:
                writer.write('{}\t{}\n'.format(id, id2fname[id]))
        with open(out_file_caption.format(i), 'wt') as writer:
            for id in id2caption:
                writer.write('{}\t{}\n'.format(id, id2caption[id]))

