# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.

import random
from collections import Counter

STUDENT1={'name': 'Coral Kuta',
         'ID': 'CORAL_ID'}
STUDENT2={'name': 'Daniel Bronfman ',
         'ID': 'DANIEL_ID '}

def read_data(fname):
    data = []
    file = open(fname, encoding='utf-8')
    for line in file:
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    file.close()
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    return [c for c in text]

fc_bi = Counter()
fc_uni = Counter()

TRAIN_BI = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
TRAIN_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("train")]
DEV_BI = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]
DEV_UNI = [(l, text_to_unigrams(t)) for l, t in read_data("dev")]

for l, feats in TRAIN_BI:
    fc_bi.update(feats)

for l, feats in TRAIN_UNI:
    fc_uni.update(feats)

# 600 most common bigrams and unigrams in the training set.
vocab_bi = set([x for x, c in fc_bi.most_common(810)])
vocab_uni = set([x for x, c in fc_uni.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN_BI]))))}

# feature strings (bigrams) to IDs
F2I_BI = {f: i for i, f in enumerate(list(sorted(vocab_bi)))}
F2I_UNI = {f: i for i, f in enumerate(list(sorted(vocab_uni)))}