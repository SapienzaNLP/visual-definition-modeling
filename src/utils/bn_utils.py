from nltk.corpus import wordnet as wn
from tqdm import tqdm
from typing import List, Dict
import _pickle as pkl


def get_cc2bn(input_file):
    cc2bn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            target_bn = fields[0].split('::')[0]
            lexeme1 = fields[0].split('::')[1]
            lexeme2 = fields[1]
            cc = (lexeme1, lexeme2)
            if cc in cc2bn:
                cc2bn.pop(cc)
                continue
            cc2bn[cc] = target_bn
    return cc2bn


def load_cooccurrences(input_file):
    coocc = dict()
    pos_dict = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'}
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')

            lemma_fields_1 = fields[0].split('#')
            lemma_fields_2 = fields[1].split('#')

            lemma_1 = lemma_fields_1[0] + '#' + pos_dict[lemma_fields_1[1]]
            lemma_2 = lemma_fields_2[0] + '#' + pos_dict[lemma_fields_2[1]]

            occs = coocc.get(lemma_1, dict())
            occs[lemma_2] = float(fields[2])
            coocc[lemma_1] = occs

            occs = coocc.get(lemma_2, dict())
            occs[lemma_1] = float(fields[2])
            coocc[lemma_2] = occs

    return coocc


def read_bn2wnoffset(input_file):
    bn2wnoffset = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2wnoffset[fields[0]] = fields[1]
    return bn2wnoffset


def read_wnoffset2bn(input_file):
    wnoffset2bn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            for f in fields[1:]:
                wnoffset2bn[f] = fields[0]
    return wnoffset2bn


def get_bn2lemma(input_file, lang='en'):
    bn2lemma = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip('\n').lower().split('\t')
            for bn in fields[1:]:
                if bn not in bn2lemma:
                    bn2lemma[bn] = set()
                lemma_fields = fields[0].split('#')  # if lang == 'en' else fields[0].split('@#*')

                bn2lemma[bn].add(lemma_fields[0].lower())
    return bn2lemma


def get_lemma2bn(input_file, pos=None, lang='en'):
    lemma2bn = dict()
    pos = ['n', 'r', 'v', 'a'] if not pos else pos
    pos = [p.lower() for p in pos]
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().lower().split('\t')
            lemma_fields = fields[0].replace('@#*', '#').split("#") # if lang == 'en' else fields[0].split('@#*')
            if lemma_fields[-1] not in pos:
                continue
            bns = lemma2bn.get(lemma_fields[0], list())
            for bn in fields[1:]:
                if bn not in bns:
                    bns.append(bn)
            lemma2bn[lemma_fields[0]] = bns
    return lemma2bn


def get_lemma2bn_pos(input_file):
    lemma2bn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().lower().split('\t')
            lemma_fields = fields[0].replace('@#*', '#')
            bns = lemma2bn.get(lemma_fields, list())
            for bn in fields[1:]:
                if bn not in bns:
                    bns.append(bn)
            lemma2bn[lemma_fields] = bns
    return lemma2bn


def get_bn2pages(input_file):
    bn2page = dict()
    bn2redi = dict()
    bn2nas = dict()
    with open(input_file, 'r', encoding="latin-1") as lines:
        for line in lines:
            fields = line.rstrip("\n").split("\t")
            bn = fields[0]
            page = fields[2].lower()
            if page.startswith("\"") and page.endswith("\""):
                page = page[1:-1]
            if page.endswith(";"):
                page = page[:-1]
            page = page.replace(" ", "_")
            page_fields = page.split(";_")
            for p in page_fields:
                if p.startswith("redi:"):
                    if bn not in bn2redi:
                        bn2redi[bn] = list()
                    bn2redi[bn].append(p[5:])
                elif p.startswith("nasari:"):
                    if bn not in bn2nas:
                        bn2nas[bn] = list()
                    bn2nas[bn].append(p[7:])
                else:
                    if bn not in bn2page:
                        bn2page[bn] = list()
                    bn2page[bn].append(p)
    return bn2page, bn2redi, bn2nas


def get_sense_distribution(word, pos, wn2bn):
    sense_distr = list()
    if pos == 'N':
        target_pos = wn.NOUN
    else:
        target_pos = wn.NOUN
    for syn in wn.synsets(word, pos=target_pos):
        lemma = syn.lemmas()[0]
        bn = wn2bn[lemma.key()]
        sense_distr.append(bn)
    return sense_distr


def read_wn2bn(input_file):
    wn2bn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            for wn in fields[1:]:
                wn2bn[wn] = fields[0]
    return wn2bn


def read_wn2bn_list(input_file):
    wn2bn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            for wn in fields[1:]:
                bns = wn2bn.get(wn, set())
                bns.add(fields[0])
                wn2bn[wn] = bns
    return wn2bn


def read_bn2wn(input_file):
    bn2wn = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip('\n').split('\t')
            bn2wn[fields[0]] = fields[1:]
    return bn2wn


def get_lemma2sensekey(wn2bn):
    lemma2sensekey = dict()
    pos_dict = {'1': 'n', '2': 'v', '3': 'a', '4': 'r', '5': 'a'}
    for wn in wn2bn:
        lemma = wn.split('%')
        lemma_pos = lemma[0] + '#' + pos_dict[lemma[1][0]]
        wns = lemma2sensekey.get(lemma_pos, set())
        wns.add(wn)
        lemma2sensekey[lemma_pos] = wns
    return lemma2sensekey


def read_sense2index(input_file):
    sense2index = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split()
            sense2index[fields[0]] = int(fields[2])
    return sense2index


def get_word2sensedistr(testwords, pos, wn2bn):
    word2sensedistr = dict()
    for word in testwords:
        word2sensedistr[word] = get_sense_distribution(word, pos, wn2bn)
    return word2sensedistr


def compute_sense_distr_wn(lemma, lemma2bn):
    sense_distr = list()
    j = 0
    for bn in lemma2bn[lemma]:
        sense_distr.append(bn)
        j += 1
    return sense_distr


def check_wordnet(bn):
    bnid = int(bn[3:-1])
    return bnid < 117660 or bnid == 14866890


def get_bn2srelated(bn2srelated, sr_threshold, nasari_vec_set=None):
    bn2page_sr = dict()
    for bn in bn2srelated:
        if nasari_vec_set and bn not in nasari_vec_set:
            # CHECK !!!
            continue
        sr_pages = dict([(s, v) for s, v in bn2srelated[bn].items() if 1. > bn2srelated[bn][s] > 0.15])
        sr_pages = sorted(sr_pages, key=lambda x: sr_pages[x], reverse=True)[:sr_threshold+1]
        bn2page_sr[bn] = sr_pages
    return bn2page_sr


def get_bn2hypernyms(in_file, wn=True):
    bn2hyp = dict()
    with open(in_file, 'rt') as lines:
        for line in tqdm(lines, desc="Loading hypernyms"):
            fields = line.rstrip().split('\t')
            hyps = set()
            target_bn = fields[0]
            if not wn or check_wordnet(target_bn):
                for bn in fields[1:]:
                    if not wn or check_wordnet(bn):
                        hyps.add(bn)
                bnhyps = bn2hyp.get(target_bn, set())
                bnhyps.update(hyps)
                bn2hyp[target_bn] = bnhyps
    return bn2hyp


def create_bnlist_from_wordlist(files: List[str], outfile: str, lemma2bn: Dict[str, List]):
    words = set()
    missing = set()
    for f in files:
        with open(f) as reader:
            words.update([w.strip().lower().replace(" ", "_") for w in reader])
    all_bns = set()
    for w in words:
        if "dosis" in w:
            print()
        if w[-2] != "#":  ##no postag attached
            added = False
            for p in "nvra":
                bn = lemma2bn.get("{}#{}".format(w, p).lower().replace(" ", "_"), None)
                if bn is not None:
                    all_bns.update(set(bn))
                    added = True
            if not added: missing.add(w)
        else:
            bn = lemma2bn.get(w, None)
            if bn is not None:
                all_bns.update(set(bn))
            else: missing.add(w)
    with open(outfile, "wb") as writer:
        pkl.dump(all_bns, writer)
    print("missing {}".format(len(missing)))
    print("\n".join(missing))
    print("wrote {} synsets".format(len(all_bns)))


def get_bn2hyponyms(in_file, wn=True):
    bn2hyp = dict()
    with open(in_file, 'rt') as lines:
        for line in tqdm(lines, desc="Loading hyponyms"):
            fields = line.rstrip().split('\t')
            target_bn = fields[0]
            if not wn or check_wordnet(target_bn):
                for bn in fields[1:]:
                    if not wn or check_wordnet(bn):
                        bnhyps = bn2hyp.get(bn, set())
                        bnhyps.add(target_bn)
                        bn2hyp[bn] = bnhyps
    return bn2hyp


def get_bn2bn_gloss(in_file):
    bn2bn_gloss = dict()
    with open(in_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2bn_gloss[fields[0]] = set(fields[1:])
    return bn2bn_gloss


def get_bn2srelated_file(input_file, target_bns=None):
    bn2srelated = dict()
    with open(input_file) as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            if target_bns and fields[0] not in target_bns:
                continue
            bn2srelated[fields[0]] = set(fields[1:])
    return bn2srelated


def load_glosses(in_file, langs, target_bns=None):
    langs = [l.upper() for l in langs]
    bn2glosses = dict()
    with open(in_file) as lines:
        for line in tqdm(lines, desc="loading glosses"):
            fields = line.rstrip().split("\t")
            bnid = fields[0]
            if target_bns and 'bn:'+bnid not in target_bns:
                continue
            lang_and_glosses = list(map(lambda elem: list(reversed(elem.strip().split("::"))),
                                        filter(lambda elem: elem.split("::")[1] in langs, fields[1:])))

            lang2glosses = dict()
            for l in lang_and_glosses:
                glosses = lang2glosses.get(l[0], set())
                if not l[1].strip().endswith("."):
                    l[1] = l[1].strip() + "."
                glosses.add(l[1])
                lang2glosses[l[0]] = glosses

            bn2glosses[bnid] = lang2glosses
    return bn2glosses


def get_bn2drelated(input_file):
    bn2drelated = dict()
    with open(input_file, 'rt') as lines:
        for line in lines:
            fields = line.rstrip().split('\t')
            bn2drelated[fields[0]] = set(fields[1:])
    return bn2drelated


def trim_bn2drelated(bn2drelated, bn2lemma, lemma2bn):
    bn2drelated_trimmed = dict()
    for bn in bn2drelated:
        lemmas = bn2lemma[bn]
        all_target_bns = set()
        for l in lemmas:
            for b in lemma2bn[l+'#'+bn[-1]]:
                if b != bn:
                    all_target_bns.add(b)
        drelated = set()
        for b in bn2drelated[bn]:
            if not any([b in bn2drelated[s] for s in all_target_bns if s in bn2drelated]) and not any([bn2lemma[b].intersection(bn2lemma[s]) for s in bn2drelated[bn] if s != b and b[-1] == s[-1]]):
                drelated.add(b)
        if len(drelated):
            bn2drelated_trimmed[bn] = drelated
    return bn2drelated_trimmed


def get_bn2syntagnet(input_file):
    bn2syntagnet = dict()
    with open(input_file) as lines:
        for line in lines:
            bn1, bn2 = line.rstrip().split()
            bns = bn2syntagnet.get(bn1, set())
            bns.add(bn2)
            bn2syntagnet[bn1] = bns
            bns = bn2syntagnet.get(bn2, set())
            bns.add(bn1)
            bn2syntagnet[bn2] = bns
    return bn2syntagnet


def load_ppr(input_file, nouns=False):
    ppr = dict()
    with open(input_file, 'rt') as lines:
        for line in tqdm(lines, desc="Loading ppr"):
            fields = line.rstrip().split('\t')
            if nouns:
                ppr[fields[0]] = dict([(t.split('_')[0], float(t.split('_')[1])) for t in fields[1:] if t.split('_')[0].endswith('n')])
            else:
                ppr[fields[0]] = dict([(t.split('_')[0], float(t.split('_')[1])) for t in fields[1:]])
    return ppr

