from __future__ import annotations

import math
from typing import Union

from plagdef import util
from plagdef.model.models import Seed
from plagdef.model.pipeline.preprocessing import Sentence, Document


class SeedFinder:
    def __init__(self, min_cos_sim: float, min_dice_sim: float):
        self._min_cos_sim = min_cos_sim
        self._min_dice_sim = min_dice_sim

    def seed(self, doc1: Document, doc2: Document) -> set[Seed]:
        _vectorize_sents(doc1, doc2)
        seeds = set()
        for doc1_sent in doc1.sents():
            for doc2_sent in doc2.sents():
                seed = self._match(doc1_sent, doc2_sent)
                if seed:
                    seeds.add(seed)
        return seeds

    def _match(self, sent1: Sentence, sent2: Sentence) -> Union[Seed, None]:
        cos_sim = util.cos_sim(sent1.tf_isf_bow, sent2.tf_isf_bow)
        dice_sim = util.dice_sim(sent1.tf_isf_bow, sent2.tf_isf_bow)
        if cos_sim > self._min_cos_sim and dice_sim > self._min_dice_sim:
            return Seed(sent1, sent2, cos_sim, dice_sim)


def _vectorize_sents(doc1: Document, doc2: Document):
    """
    Compute the tf-isf = tf x ln(N/sf), N being the number of sentences in corpus
    and sf the number of sentences containing the term
    """
    sf = doc1.vocab + doc2.vocab
    doc1_sents, doc2_sents = list(doc1.sents()), list(doc2.sents())
    N = len(doc1_sents) + len(doc2_sents)
    for sent in doc1_sents:
        for lemma in sent.bow:
            sent.tf_isf_bow[lemma] = sent.bow[lemma] * math.log(N / float(sf[lemma]))
    for sent in doc2_sents:
        for lemma in sent.bow:
            sent.tf_isf_bow[lemma] = sent.bow[lemma] * math.log(N / float(sf[lemma]))
