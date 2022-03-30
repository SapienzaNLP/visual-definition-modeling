from transformers import LxmertTokenizer

from src.modeling.model import Lxmert_LM
from src.utils import frcnn_utils
from src.utils.frcnn_utils import Config
from src.utils.modeling_frcnn import GeneralizedRCNN
from src.utils.processing_image import Preprocess

if __name__ == "__main__":
    model = Lxmert_LM().cuda()
    frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg).cuda()

    image_preprocess = Preprocess(frcnn_cfg)

    URL = "https://vqa.cloudcv.org/media/test2014/COCO_test2014_000000262567.jpg"
    OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
    ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
    GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
    VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"

    objids = frcnn_utils.get_data(OBJ_URL)
    attrids = frcnn_utils.get_data(ATTR_URL)
    gqa_answers = frcnn_utils.get_data(GQA_URL)
    vqa_answers = frcnn_utils.get_data(VQA_URL)

    lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images.cuda(),
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt"
    )

    test_sentence = "this is a <define>stupid<define> test sentence."
    output_sentence = "lacking intelligence"
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    inputs = lxmert_tokenizer(
        test_sentence,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    decoder_input = lxmert_tokenizer(
        output_sentence,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    logits = model(input_ids=inputs.input_ids.cuda(),
                   attention_mask=inputs.attention_mask.cuda(),
                   token_type_ids=inputs.token_type_ids.cuda(),
                   visual_feats=features.cuda(),
                   visual_pos=normalized_boxes.cuda(),
                   decoder_input_ids=decoder_input.input_ids.cuda(),
                   decoder_padding_mask=decoder_input.attention_mask.cuda(),
                   labels=decoder_input.input_ids.cuda()
                   )
