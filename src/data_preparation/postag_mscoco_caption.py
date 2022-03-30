from xml.dom import minidom

import spacy
import argparse
from xml.etree import ElementTree as etree
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mscoco_caption", type=str,
                        default='/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco_caption.{}.txt')
    parser.add_argument('--out_file_xml', type=str,
                        default="/home/bianca/PycharmProjects/MultimodalGlosses/data/out/mscoco.caption.{}.data.xml")

    args = parser.parse_args()
    mscoco_caption = args.mscoco_caption
    out_file_xml = args.out_file_xml

    content_pos = {'NOUN':'NOUN', 'VERB':'VERB', 'PROPN':'NOUN', 'ADJ':'ADJ', 'ADV':'ADV'}

    for i in ['train', 'val']:
        id2caption = dict()
        with open(mscoco_caption.format(i), 'rt') as lines:
            for line in lines:
                fields = line.rstrip().split('\t')
                if len(fields) == 2:
                    id2caption[fields[0]] = fields[1]

        nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'textcat'])
        corpus = etree.Element('corpus', {'lang': 'EN', 'source': 'mscoco'})
        text = etree.SubElement(corpus, 'text', {'id': 'd000', 'source': 'mscoco'})

        for j, id in tqdm(enumerate(id2caption)):
            sentence_id = 'd000.s' + str(j)
            sentence = etree.SubElement(text, 'sentence', {'id': sentence_id})
            doc = nlp(id2caption[id])
            for k, tok in enumerate(doc):
                if tok.pos_ in content_pos:
                    t = etree.SubElement(sentence, 'instance', {'id': sentence_id + 't{}'.format(k),
                                                                  'lemma':tok.lemma_,
                                                                  'pos':content_pos[tok.pos_]})
                    t.text = tok.text
                else:
                    t = etree.SubElement(sentence, 'wf', {'lemma': tok.lemma_,
                                                            'pos': tok.pos_})
                    t.text = tok.text

        t = minidom.parseString(etree.tostring(corpus, encoding='utf-8')).toprettyxml(encoding='utf-8')
        tree1 = etree.ElementTree(etree.fromstring(t))
        tree1.write(out_file_xml.format(i), encoding='utf-8', xml_declaration=True)