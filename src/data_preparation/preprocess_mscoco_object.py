import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscoco_ann_dir", type=str,
                        default='/media/bianca/f1f90d67-e33e-4e88-a6c3-d85a075682da/mscoco/annotations/instances_{}2017.json')
    parser.add_argument('--out_file', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_object.{}.txt")

    args = parser.parse_args()
    mscoco_ann_dir = args.mscoco_ann_dir
    out_file = args.out_file

    for i in ['train', 'val']:
        id2obj = dict()
        id2categories = dict()
        json_data = json.load(open(mscoco_ann_dir.format(i), 'r'))
        for j in json_data['categories']:
            id2categories[j['id']] = (j['name'], j['supercategory'])
        for j in json_data['annotations']:
            img = j['image_id']
            objects = id2obj.get(img, dict())
            objects[j['id']] = (j['bbox'], j['category_id'])
            id2obj[img] = objects
        with open(out_file.format(i), 'wt') as writer:
            for id in id2obj:
                for k in id2obj[id]:
                    writer.write('{}\t{}\t{}\t{}\n'.format(id, k, '\t'.join(id2categories[id2obj[id][k][1]]),
                                                       '\t'.join([str(x) for x in id2obj[id][k][0]])))

