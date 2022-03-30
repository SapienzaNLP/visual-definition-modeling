from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument("--definition_file", type=str, required=True)
    argparse.add_argument("--generation_file", type=str, required=True)
    args = argparse.parse_args()
    definition_file = args.definition_file
    generation_file = args.generation_file

    coco = COCO(definition_file)
    cocoRes = coco.loadRes(generation_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

