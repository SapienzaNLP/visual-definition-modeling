from typing import List, Tuple, Dict

import os
import matplotlib.pyplot as plt
from matplotlib import image
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import numpy as np
from PIL import Image


def load_semcor(path):
    all_sentences = list()
    sids = list()
    with open(path) as lines:
        sentence_id = None
        words = list()
        sentence = ""
        sid = ""
        for line in lines:
            fields = line.strip().split("\t")
            local_sentence = fields[1]
            local_id = ".".join(fields[0].split(".")[:-1])
            if sentence_id is None:
                sentence_id = local_id
            if local_id != sentence_id:
                all_sentences.append((sentence, words))
                sids.append(sid)
                sentence = ""
                words = list()
            sentence_id = local_id
            sentence = local_sentence
            sid = local_id
            words.append((fields[2], fields[4]))

    return dict([reversed(x) for x in enumerate(sids)]), all_sentences

from textwrap import wrap

def print_sentence_images(sentence: str, tokens: List[str], synset2img: Dict[str, str]):
    images = list()
    for word, synset in tokens:
        imgpath = synset2img.get(synset, None)
        if imgpath is None:
            continue
        synset_image = Image.open(imgpath) #image.imread(imgpath)
        synset_image.thumbnail((128, 128), Image.ANTIALIAS)
        images.append((word ,synset_image))

    fig = plt.figure(figsize=(6., 6.))
    fig.suptitle("\n".join(wrap(sentence, 60)))

    nrows = len(images) // 3 + 1
    if len(images) == 0:
        print("no images found!")
        return
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, min(3, len(images))),  # creates 2x2 grid of axes
                     axes_pad=1,  # pad between axes in inch.
                     )

    for ax, (word, im) in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.set_title(word)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax.imshow(im)
    plt.show()


def load_synset2img(path, img_dir_path):
    synset2img = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split("\t")
            synset2img[fields[0]] = os.path.join(img_dir_path, fields[1])
    return synset2img


if __name__ == "__main__":
    semcor = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data/in/semcor.data.dm.txt"
    synset2imgpath = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data/in/semcor_babelpic/synset2filename.txt"
    img_dir_path = "/home/tommaso/dev/PycharmProjects/MultimodalGlosses/data/in/semcor_babelpic"
    synset2img = load_synset2img(synset2imgpath, img_dir_path)
    sid2idx, all_sentences = load_semcor(semcor)

    while True:
        inp = input("input a sentence id or type enter")
        if len(inp.strip()):
            idx = sid2idx[inp.strip()]
        else:
            idx = random.randint(0, len(all_sentences) - 1)
        sentence, tokens = all_sentences[idx]
        print_sentence_images(sentence, tokens, synset2img)
