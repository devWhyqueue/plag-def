from __future__ import annotations

import logging
from itertools import combinations, product

from tqdm import tqdm

from plagdef.model import matching
from plagdef.model.matching import PipeComponents, VerbatimMatcher
from plagdef.model.models import Document, DocumentPairMatches
from plagdef.model.pipeline.extension import ClusterBuilder
from plagdef.model.pipeline.filtering import ClusterFilter
from plagdef.model.pipeline.preprocessing import Preprocessor
from plagdef.model.pipeline.seeding import SeedFinder
from plagdef.util import parallelize

log = logging.getLogger(__name__)


class DocumentMatcher:
    def __init__(self, config: dict):
        self._preprocessor = Preprocessor(config['min_sent_len'], config['rem_stop_words'])
        self._seeder = SeedFinder(config['min_cos_sim'], config['min_dice_sim'])
        self._verbatim_matcher = VerbatimMatcher(config['min_verbatim_match_char_len'])
        self._intelligent_cb = ClusterBuilder(config['adjacent_sents_gap'], config['min_adjacent_sents_gap'],
                                              config['min_sent_number'], config['min_cluster_cos_sim'])
        self._summary_cb = ClusterBuilder(config['adjacent_sents_gap_summary'], config['min_adjacent_sents_gap'],
                                          config['min_sent_number'], config['min_cluster_cos_sim'])
        self._cluster_filter = ClusterFilter(config['min_cluster_char_len'])

    def preprocess(self, lang: str, docs: set[Document], common_docs=None):
        self._preprocessor.preprocess(lang, docs, common_docs)

    def find_matches(self, docs: set[Document], archive_docs=None) -> list[DocumentPairMatches]:
        doc_combs = set(combinations(docs, 2))
        if archive_docs:
            doc_combs.update(product(docs, archive_docs))
        return parallelize(self._find_matches, list(doc_combs))

    def _find_matches(self, doc_combs, pos=0) -> list[DocumentPairMatches]:
        matches = []
        for doc1, doc2 in tqdm(doc_combs, desc='Matching', unit='pair', total=len(doc_combs), position=pos,
                               leave=False):
            pipe = matching.Pipeline(doc1, doc2,
                                     PipeComponents(self._seeder, self._verbatim_matcher, self._intelligent_cb,
                                                    self._summary_cb, self._cluster_filter))
            doc_pair_matches = pipe.find_matches()
            matches.append(doc_pair_matches) if len(doc_pair_matches) else None
        return matches
