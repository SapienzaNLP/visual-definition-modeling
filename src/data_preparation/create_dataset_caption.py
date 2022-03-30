import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_caption.val.txt')
    parser.add_argument("--imgid_file", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_imgid.val.txt')
    parser.add_argument('--out_file', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/in/mscoco.caption.val.dm.txt")

    args = parser.parse_args()
    input_file = args.input_file
    imgid_file = args.imgid_file
    out_file = args.out_file

    id2caption = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            id2caption[fields[0]] = fields[1]

    id2img = dict()
    with open(imgid_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            id2img[fields[0]] = fields[1]

    with open(out_file, 'wt') as writer:
        for id in id2caption:
            writer.write('caption.{}\t<s>\t<s>\t0-1\t{}\t{}\t{}\n'.format(id, id2img[id], id2caption[id], id2img[id]))