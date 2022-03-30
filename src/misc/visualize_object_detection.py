import PIL

from src.utils.modeling_frcnn import GeneralizedRCNN
from src.utils.frcnn_utils import Config, get_data
from src.utils.processing_image import Preprocess
from src.utils.visualizing_image import SingleImageViz
import numpy as np
import os
import random
from tqdm import tqdm


def showarray(a, file=None, show=True):
    a = np.uint8(np.clip(a, 0, 255))
    img = PIL.Image.fromarray(a)
    if show:
        img.show()
    if file is not None:
        img.save(file)


if __name__ == '__main__':

    babelpic_dir = '/home/bianca/PycharmProjects/MultimodalGlosses/data/in/babelpic/babelpic-gold/images/'
    new_url = '/home/bianca/PycharmProjects/MultimodalGlosses/data/out/babelpic_objdet/'
    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    num_sample = 100
    batch_size = 2

    objids = get_data(OBJ_URL)
    attrids = get_data(ATTR_URL)

    frcnn_cfg = Config.from_pretrained('unc-nlp/frcnn-vg-finetuned')
    frcnn = GeneralizedRCNN.from_pretrained('unc-nlp/frcnn-vg-finetuned', config=frcnn_cfg).to('cuda:0')
    image_preprocess = Preprocess(frcnn_cfg)

    image_files = random.sample([os.path.join(babelpic_dir, f) for f in os.listdir(babelpic_dir)
                          if os.path.isfile(os.path.join(babelpic_dir, f))], num_sample)

    for image_batch in tqdm(range(0, len(image_files), batch_size), desc="Computing image embeddings"):

        images, sizes, scales_yx = image_preprocess(image_files[image_batch: image_batch+batch_size])

        output_dict = frcnn(images.to('cuda:0'), sizes, scales_yx=scales_yx, padding="max_detections", max_detections=frcnn_cfg.max_detections,
            return_tensors="pt")

        for img, box, obj_ids, obj_probs, attr_ids, attr_probs in zip(image_files[image_batch: image_batch+batch_size],
                                                                      output_dict.get("boxes"),
            output_dict.pop("obj_ids"),
            output_dict.pop("obj_probs"),
            output_dict.pop("attr_ids"),
            output_dict.pop("attr_probs")):

            frcnn_visualizer = SingleImageViz(img, id2obj=objids, id2attr=attrids)

            frcnn_visualizer.draw_boxes(box, obj_ids, obj_probs, attr_ids, attr_probs,)

            showarray(frcnn_visualizer._get_buffer(), file=os.path.join(new_url, img.split('/')[-1]), show=False)