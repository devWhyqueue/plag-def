from __future__ import annotations

from dataclasses import dataclass

from plagdef.model.models import Document, DocumentPairMatches, Match, MatchType, Cluster, Fragment
from plagdef.model.pipeline.extension import ClusterBuilder
from plagdef.model.pipeline.filtering import ClusterFilter
from plagdef.model.pipeline.seeding import SeedFinder


@dataclass(frozen=True)
class PipeComponents:
    seeder: SeedFinder
    verbatim_matcher: VerbatimMatcher
    intelligent_cb: ClusterBuilder
    summary_cb: ClusterBuilder
    cf: ClusterFilter


class Pipeline:
    def __init__(self, doc1: Document, doc2: Document, pipe_comps: PipeComponents):
        self._doc1 = doc1
        self._doc2 = doc2
        self._pipe_comps = pipe_comps

    def find_matches(self) -> DocumentPairMatches:
        doc_pair_matches = DocumentPairMatches(self._doc1, self._doc2)
        seeds = self._pipe_comps.seeder.seed(self._doc1, self._doc2)
        clusters = self._build_clusters(seeds, self._pipe_comps.intelligent_cb)
        verbatim_matches = intelligent_matches = summary_matches = set()
        if len(clusters):
            verbatim_matches = self._pipe_comps.verbatim_matcher.find_verbatim_matches(clusters)
            intelligent_matches = {Match.from_cluster(MatchType.INTELLIGENT, cluster) for cluster in clusters}
        summary_clusters = self._build_clusters(seeds, self._pipe_comps.summary_cb)
        if len(summary_clusters):
            sum_cluster_len_doc1, sum_cluster_len_doc2 = \
                tuple(map(sum, zip(*(cluster.char_lengths() for cluster in summary_clusters))))
            if sum_cluster_len_doc1 >= 3 * sum_cluster_len_doc2 \
                or sum_cluster_len_doc2 >= 3 * sum_cluster_len_doc1:
                summary_matches = {Match.from_cluster(MatchType.SUMMARY, cluster) for cluster in summary_clusters}
        doc_pair_matches.update({*verbatim_matches, *intelligent_matches, *summary_matches})
        return doc_pair_matches

    def _build_clusters(self, seeds, cluster_builder: ClusterBuilder):
        clusters = cluster_builder.extend(seeds)
        clusters = self._pipe_comps.cf.filter(clusters)
        return clusters


class VerbatimMatcher:
    def __init__(self, min_verbatim_match_char_len: int):
        self._min_verbatim_match_char_len = min_verbatim_match_char_len

    def find_verbatim_matches(self, clusters: set[Cluster]) -> set[Match]:
        matches = set()
        for cluster in clusters:
            cluster_matches = self._common_words(cluster)
            matches.update(VerbatimMatcher._resolve_match_overlaps(cluster_matches))
        return matches

    def _common_words(self, cluster: Cluster) -> set[Match]:
        verbatim_matches = set()
        frag1_words = [word for sent_words in [sent.words for sent in cluster.sents_doc1] for word in sent_words]
        frag2_words = [word for sent_words in [sent.words for sent in cluster.sents_doc2] for word in sent_words]
        lookup = [[0 for _ in range(len(frag2_words) + 1)] for _ in range(len(frag1_words) + 1)]
        for i in range(1, len(frag1_words) + 1):
            for j in range(1, len(frag2_words) + 1):
                if frag1_words[i - 1].text.lower() == frag2_words[j - 1].text.lower():
                    lookup[i][j] = lookup[i - 1][j - 1] + 1
                    match_char_len = sum([len(word) for word in frag1_words[i - lookup[i][j]:i]])
                    if match_char_len >= self._min_verbatim_match_char_len:
                        frag1_first, frag2_first = i - lookup[i][j], j - lookup[i][j]
                        # Include punctuation (mostly periods) if exists
                        punct = VerbatimMatcher._include_punct(cluster, frag1_words[i - 1].end_char,
                                                               frag2_words[j - 1].end_char)
                        frag1 = Fragment(frag1_words[frag1_first].start_char, frag1_words[i - 1].end_char + punct,
                                         cluster.doc1)
                        frag2 = Fragment(frag2_words[frag2_first].start_char, frag2_words[j - 1].end_char + punct,
                                         cluster.doc2)
                        verbatim_matches.add(Match(MatchType.VERBATIM, frag1, frag2))
        return verbatim_matches

    @classmethod
    def _resolve_match_overlaps(cls, matches: set[Match]) -> set[Match]:
        non_ol_matches = set()
        for match in sorted(matches, key=len, reverse=True):
            if not any(match.overlaps_with(non_ol_match) for non_ol_match in non_ol_matches):
                non_ol_matches.add(match)
        return non_ol_matches

    @classmethod
    def _include_punct(cls, cluster: Cluster, frag1_end_char: int, frag2_end_char: int) -> int:
        text_ends_with_word = frag1_end_char == len(cluster.doc1.text) or frag2_end_char == len(cluster.doc2.text)
        if not text_ends_with_word and cluster.doc1.text[frag1_end_char] == cluster.doc2.text[frag2_end_char] \
            and not cluster.doc1.text[frag1_end_char].isspace():
            return 1
        return 0
