from collections import Counter
from unittest.mock import patch

import pytest

from plagdef.model.models import Document, Sentence, Cluster, Fragment, Word, RatedCluster, Match, \
    DocumentPairMatches, \
    DifferentDocumentPairError, SameDocumentError, MatchType
from plagdef.tests.model.pipeline.test_extension import _create_seeds


def test_document_sents():
    doc = Document('doc', 'path/to/doc', 'Some text.')
    doc.add_sent(Sentence(0, 10, Counter(), doc))
    assert len(list(doc.sents(include_common=True))) == 1
    assert list(doc.sents(include_common=True))[0].text == 'Some text.'


def test_document_remove_sent():
    doc = Document('doc', 'path/to/doc', 'Some text.')
    sent = Sentence(0, 10, Counter(), doc)
    doc.add_sent(sent)
    doc.remove_sent(sent)
    assert len(list(doc.sents(include_common=True))) == 0


def test_document_sents_exclude_common():
    doc = Document('doc', 'path/to/doc', 'Some text. A common sent.')
    doc.add_sent(Sentence(0, 10, Counter(), doc))
    common_sent = Sentence(11, 25, Counter(), doc)
    common_sent.common = True
    doc.add_sent(common_sent)
    assert len(list(doc.sents())) == 1
    assert len(list(doc.sents(include_common=True))) == 2
    assert list(doc.sents(include_common=True))[0].text == 'Some text.'
    assert list(doc.sents(include_common=True))[1].text == 'A common sent.'


def test_documents_are_equal():
    doc1 = Document('doc', 'path/to/doc', 'Some text.')
    doc2 = Document('doc', 'path/to/other/doc', 'Some text.')
    assert doc1 == doc2


def test_documents_are_the_same_if_same_name_and_text():
    docs = set()
    docs.add(Document('doc', 'path/to/doc', 'abc'))
    docs.add(Document('doc', 'path/to/other/doc', 'abc'))
    assert len(docs) == 1


def test_document_sents_are_ordered_by_start_char():
    doc = Document('doc', 'path/to/doc', '')
    doc.add_sent(Sentence(5, -1, Counter(), doc))
    doc.add_sent(Sentence(3, -1, Counter(), doc))
    doc.add_sent(Sentence(7, -1, Counter(), doc))
    assert [sent.start_char for sent in doc.sents(include_common=True)] == [3, 5, 7]


def test_fragment_length():
    frag = Fragment(0, 15, Document('doc', 'path/to/doc', ''))
    assert len(frag) == 15


def test_fragment_overlaps():
    doc = Document('doc', 'path/to/doc', '')
    frag1 = Fragment(0, 15, doc)
    frag2 = Fragment(4, 8, doc)
    assert frag1.overlaps_with(frag2) and frag2.overlaps_with(frag1)


def test_fragment_overlaps_with_different_docs():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    frag1 = Fragment(0, 15, doc1)
    frag2 = Fragment(14, 16, doc2)
    assert not frag1.overlaps_with(frag2) and not frag2.overlaps_with(frag1)


def test_sentence_idx():
    doc = Document('doc', 'path/to/doc', '')
    sent = Sentence(3, -1, Counter(), doc)
    doc.sents(include_common=True).add(Sentence(4, -1, Counter(), doc))
    doc.sents(include_common=True).add(sent)
    doc.sents(include_common=True).add(Sentence(0, -1, Counter(), doc))
    assert sent.idx == 1


def test_sentences_are_equal():
    doc = Document('doc', 'path/to/doc', '')
    sent1 = Sentence(3, 17, Counter(), doc)
    sent2 = Sentence(3, 17, Counter({'xyz': 2}), doc)  # unrealistic
    sent3 = Sentence(4, 17, Counter(), doc)
    assert sent1 == sent2, sent1 != sent3


def test_sentences_are_the_same_if_doc_and_start_char_are_equal():
    doc = Document('doc', 'path/to/doc', '')
    sents = set()
    sents.add(Sentence(3, 17, Counter(), doc))
    sents.add(Sentence(3, 17, Counter({'xyz': 2}), doc))  # unrealistic
    assert len(sents) == 1


def test_sentence_adjacent_to():
    sent1 = _create_sent('doc1', 0)
    sent2 = _create_sent('doc2', 5)
    adj = sent1.adjacent_to(sent2, 4)
    assert adj


def test_sentence_adjacent_to_with_gap_over_th():
    sent1 = _create_sent('doc1', 0)
    sent2 = _create_sent('doc2', 6)
    # Five sentences in-between
    adj = sent1.adjacent_to(sent2, 4)
    assert not adj


def test_word_equals():
    sent = Sentence(0, 15, Counter(), Document('doc', 'path/to/doc', ''))
    word1, word2, word3 = Word(0, 6, sent), Word(0, 6, sent), Word(6, 15, sent)
    assert word1 == word2
    assert word1 != word3 and word2 != word3


def test_clusters_are_equal_if_same_seeds():
    cluster1 = Cluster(_create_seeds([(0, 1), (10, 9)]))
    cluster2 = Cluster(_create_seeds([(10, 9), (0, 1)]))
    assert cluster1 == cluster2


def test_clusters_are_the_same_if_same_seeds():
    clusters = set()
    clusters.add(Cluster(_create_seeds([(0, 1), (10, 9)])))
    clusters.add(Cluster(_create_seeds([(10, 9), (0, 1)])))
    assert len(clusters) == 1


def test_cluster_contains_sents_from_both_docs():
    cluster = Cluster(_create_seeds([(3, 1), (10, 9)]))
    assert len(cluster.sents_doc1) == 8
    assert len(cluster.sents_doc2) == 9


def test_cluster_tf_isf_bow():
    # doc1: 'This is an awesome document. All of these bows are combined. Even this last one.'
    # doc2: 'This is another document. Just for good measure. As always.'
    # Just for this example. In reality the last two sentences are too different.
    cluster = Cluster(_create_seeds([(0, 0), (1, 1), (2, 2)]))
    # Simulate the preprocessing for doc1
    cluster.sents_doc1 = (Sentence(0, -1, Counter(), cluster.doc1),
                          Sentence(1, -1, Counter(), cluster.doc1),
                          Sentence(2, -1, Counter(), cluster.doc1))
    cluster.sents_doc1[0].tf_isf_bow = {'this': 0.22, 'be': 0.51, 'a': 1.60, 'awesome': 1.60, 'document': 0.91}
    cluster.sents_doc1[1].tf_isf_bow = {'all': 1.60, 'of': 1.60, 'this': 0.22, 'bow': 1.60, 'be': 0.51, 'combine': 1.60}
    cluster.sents_doc1[2].tf_isf_bow = {'even': 1.60, 'this': 0.22, 'last': 1.60, 'one': 1.60}
    tf_isf_bow_doc1 = cluster._tf_isf_bow(doc1_sents=True)
    assert tf_isf_bow_doc1 == {'this': 0.66, 'be': 1.02, 'a': 1.6, 'awesome': 1.6, 'document': 0.91,
                               'all': 1.6, 'of': 1.6, 'bow': 1.6, 'combine': 1.6, 'even': 1.6, 'last': 1.6, 'one': 1.6}


def test_clusters_overlap():
    cluster1 = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster2 = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    assert cluster1.overlaps_with(cluster2)
    assert cluster2.overlaps_with(cluster1)


def test_clusters_do_not_overlap_if_they_only_share_a_sentence_in_first_doc():
    cluster1 = Cluster(_create_seeds([(0, 1), (4, 5)]))
    cluster2 = Cluster(_create_seeds([(3, 6), (7, 7)]))
    assert not cluster1.overlaps_with(cluster2)
    assert not cluster2.overlaps_with(cluster1)


def test_clusters_do_not_overlap_if_they_only_share_a_sentence_in_second_doc():
    cluster1 = Cluster(_create_seeds([(0, 1), (4, 5)]))
    cluster2 = Cluster(_create_seeds([(6, 5), (7, 7)]))
    assert not cluster1.overlaps_with(cluster2)


def test_cluster_fragment_similarity():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    # Calculating sim_{a.sents_doc2}(a.sents_doc1 ∩ b.sents_doc1)
    # fragment_sents: 4, 5, 6 (a.sents_doc1 ∩ b.sents_doc1)
    # cluster_sents: 0, 1, 2, 3, 4 (a.sents_doc2)
    # cos_sims for fragment_sent[0] for each cluster_sent: [0, 0, 0, 0.4, 0]
    # cos_sims for fragment_sent[1] for each cluster_sent: [0.6, 0, 0, 0, 0]
    # cos_sims for fragment_sent[2] for each cluster_sent: [0, 0, 0, 0, 0.5]
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.side_effect = [0, 0, 0, 0.4, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.5]
        sim = cluster_a._fragment_similarity(cluster_a.sents_doc1[4:7], cluster_a.sents_doc2)
    assert sim == 0.5


def test_cluster_rate_in_respect_to():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    # Calculating q_{b}(a)
    # Then sim_{a.sents_doc2}(a.sents_doc1 ∩ b.sents_doc1) = 0.5
    # fragment_sents: 4, 5, 6 (a.sents_doc1 ∩ b.sents_doc1)
    # cluster_sents: 0, 1, 2, 3, 4 (a.sents_doc2)
    # cos_sims for fragment_sent[0] for each cluster_sent: [0, 0, 0, 0.4, 0]
    # cos_sims for fragment_sent[1] for each cluster_sent: [0.6, 0, 0, 0, 0]
    # cos_sims for fragment_sent[2] for each cluster_sent: [0, 0, 0, 0, 0.5]
    # and sim_{a.sents_doc2}(a.sents_doc1 / (a.sents_doc1 ∩ b.sents_doc1)) = 0.4
    # fragment_sents: 0, 1, 2, 3 (a.sents_doc1 / (a.sents_doc1 ∩ b.sents_doc1))
    # cluster_sents: 0, 1, 2, 3, 4 (a.sents_doc2)
    # cos_sims for fragment_sent[0] for each cluster_sent: [0, 0, 0, 0.4, 0]
    # cos_sims for fragment_sent[1] for each cluster_sent: [0.2, 0, 0, 0, 0]
    # cos_sims for fragment_sent[2] for each cluster_sent: [0, 0, 0, 0, 0.1]
    # cos_sims for fragment_sent[3] for each cluster_sent: [0, 0.9, 0, 0, 0]
    # => q_{b}(a) = 0.5 + (1 - 0.5) * 0.4 = 0.7, size: 7
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.side_effect = [0, 0, 0, 0.4, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.5,
                               0, 0, 0, 0.4, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.9, 0, 0, 0]
        rated_cluster = cluster_a._rate_with_respect_to(cluster_b, first_doc_susp=True)
    assert rated_cluster.quality == 0.7
    assert rated_cluster.size == 7


def test_cluster_best_variant():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    # Picking a variant a (doc1 is susp and doc2 is src) or b (doc2 is susp and doc1 is src)
    # a: q_{b}(a) = 0.5 + (1 - 0.5) * 0.4 = 0.7, size: 7
    # (for more details take a look at test_cluster_rate_in_respect_to())
    # b:
    # Calculating q_{b}(a)
    # Then sim_{a.sents_doc1}(a.sents_doc2 ∩ b.sents_doc2) = 0.6
    # fragment_sents: 2, 3, 4 (a.sents_doc2 ∩ b.sents_doc2)
    # cluster_sents: 0, 1, 2, 3, 4, 5, 6 (a.sents_doc1)
    # cos_sims for fragment_sent[0] for each cluster_sent: [0.5, 0, 0, 0.4, 0, 0, 0]
    # cos_sims for fragment_sent[1] for each cluster_sent: [0.6, 0.8, 0, 0, 0, 0, 0]
    # cos_sims for fragment_sent[2] for each cluster_sent: [0, 0, 0, 0, 0.5, 0, 0]
    # and sim_{a.sents_doc1}(a.sents_doc2 / (a.sents_doc2 ∩ b.sents_doc2)) = 0.3
    # fragment_sents: 0, 1 (a.sents_doc2 / (a.sents_doc2 ∩ b.sents_doc2))
    # cluster_sents: 0, 1, 2, 3, 4, 5, 6 (a.sents_doc1)
    # cos_sims for fragment_sent[0] for each cluster_sent: [0, 0, 0, 0.4, 0, 0, 0]
    # cos_sims for fragment_sent[1] for each cluster_sent: [0.2, 0, 0, 0, 0, 0, 0]
    # => q_{b}(a) = 0.6 + (1 - 0.6) * 0.3 = 0.72, size: 5
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.side_effect = [0, 0, 0, 0.4, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.5,
                               0, 0, 0, 0.4, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.9, 0, 0, 0,
                               0.5, 0, 0, 0.4, 0, 0, 0, 0.6, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
                               0, 0, 0, 0.4, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0]
        rated_cluster = cluster_a._best_variant(cluster_b)
    assert rated_cluster.quality == 0.72
    assert rated_cluster.size == 5


def test_cluster_best_in_respect_to():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    # Best variant of cluster a: q_{b}(a) = 0.72, size: 5
    # Best variant of cluster b: q_{a}(b) = 0.74, size: 7
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.side_effect = [0, 0, 0, 0.4, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0.5,
                               0, 0, 0, 0.4, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.9, 0, 0, 0,
                               0.5, 0, 0, 0.4, 0, 0, 0, 0.6, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
                               0, 0, 0, 0.4, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0,

                               0, 0, 0, 0.4, 0, 0, 0, 0.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5,
                               0, 0, 0, 0.4, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.9, 0, 0, 0, 0,
                               0,
                               0.5, 0, 0, 0.4, 0, 0, 0, 0.6, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
                               0, 0, 0, 0.4, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0, 0, 0, 0, 0, 0, 0.4, 0, 0,
                               0]
        rated_cluster = cluster_a.best_with_respect_to(cluster_b)
    assert rated_cluster.quality == 0.74
    assert rated_cluster.size == 7


def test_cluster_char_lengths():
    cluster = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    char_lengths = cluster.char_lengths()
    assert char_lengths[0] == 7 and char_lengths[1] == 5


def test_cluster_char_lengths_ignores_common_sents():
    seeds = _create_seeds([(0, 4), (3, 9)])
    for i in range(1, 3):
        seeds[0].sent1.doc.sents(include_common=True)[i].common = True
    for i in range(2, 4):
        seeds[0].sent2.doc.sents(include_common=True)[i].common = True
    cluster = Cluster(seeds)
    char_lengths = cluster.char_lengths()
    assert char_lengths[0] == 2 and char_lengths[1] == 4


def test_rated_cluster_equality():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    rated_cluster_a = RatedCluster(cluster_a, 0.7, 10)
    rated_cluster_b = RatedCluster(cluster_b, 0.6, 10)
    assert rated_cluster_a > rated_cluster_b and rated_cluster_a >= rated_cluster_b \
           and rated_cluster_b < rated_cluster_a and rated_cluster_b <= rated_cluster_a


def test_rated_cluster_equality_with_same_quality():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    rated_cluster_a = RatedCluster(cluster_a, 0.7, 11)
    rated_cluster_b = RatedCluster(cluster_b, 0.7, 10)
    assert rated_cluster_a > rated_cluster_b and rated_cluster_a >= rated_cluster_b \
           and rated_cluster_b < rated_cluster_a and rated_cluster_b <= rated_cluster_a


def test_rated_cluster_equality_with_high_quality():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    rated_cluster_a = RatedCluster(cluster_a, 0.998, 11)
    rated_cluster_b = RatedCluster(cluster_b, 0.999, 10)
    assert rated_cluster_a > rated_cluster_b and rated_cluster_a >= rated_cluster_b \
           and rated_cluster_b < rated_cluster_a and rated_cluster_b <= rated_cluster_a


def test_match_init_with_fragments_of_same_doc_fails():
    doc = Document('doc', 'path/to/doc', '')
    with pytest.raises(SameDocumentError):
        Match(MatchType.VERBATIM, Fragment(0, 7, doc), Fragment(13, 15, doc))


def test_match_from_cluster():
    cluster = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    match = Match.from_cluster(MatchType.VERBATIM, cluster)
    assert match == Match(MatchType.VERBATIM, Fragment(0, 7, cluster.doc1), Fragment(0, 5, cluster.doc2))


def test_match_overlaps_with():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(13, 15, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc2), Fragment(6, 9, doc1))
    assert match1.overlaps_with(match2) and match2.overlaps_with(match1)


def test_match_frag_from_doc():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(13, 15, doc2))
    frag1, frag2 = match.frag_from_doc(doc1), match.frag_from_doc(doc2)
    assert frag1.doc == doc1
    assert frag2.doc == doc2


def test_match_frag_from_doc_with_other_doc():
    doc1, doc2, doc3 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', ''), \
                       Document('doc3', 'path/to/doc3', '')
    match = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(13, 15, doc2))
    frag = match.frag_from_doc(doc3)
    assert frag is None


def test_matches_do_not_overlap_if_overlap_only_in_one_frag():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc2), Fragment(6, 9, doc1))
    assert not match1.overlaps_with(match2) and not match2.overlaps_with(match1)


def test_match_len():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match = Match(MatchType.VERBATIM, Fragment(0, 5, doc1), Fragment(0, 30, doc2))
    assert len(match) == 35


def test_doc_pair_matches_init():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    doc_pair_matches = DocumentPairMatches(doc1, doc2)
    assert doc_pair_matches._matches == {}


def test_doc_pair_matches_init_with_matches():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.INTELLIGENT, Fragment(10, 15, doc2), Fragment(6, 9, doc1))
    doc_pair_matches = DocumentPairMatches(doc1, doc2, {match1, match2})
    assert doc_pair_matches._matches == {str(MatchType.VERBATIM): {match1}, str(MatchType.INTELLIGENT): {match2}}


def test_doc_pair_matches_from_matches_with_empty_list():
    doc_pair_matches = DocumentPairMatches.from_matches([])
    assert doc_pair_matches == []


def test_doc_pair_matches_from_matches(matches):
    match_list = []
    for match_type in MatchType:
        [match_list.extend(m.list(match_type)) for m in matches]
    doc_pair_matches = DocumentPairMatches.from_matches(match_list)
    assert set(doc_pair_matches) == matches


def test_doc_pair_matches_equal_if_docs_are_the_same():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    doc_pair_matches1 = DocumentPairMatches(doc1, doc2)
    doc_pair_matches2 = DocumentPairMatches(doc2, doc1, {match})
    assert doc_pair_matches1 == doc_pair_matches2


def test_doc_pair_matches_add():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc2), Fragment(6, 9, doc1))
    doc_pair_matches = DocumentPairMatches(doc1, doc2)
    doc_pair_matches.add(match1)
    doc_pair_matches.add(match2)
    assert len(doc_pair_matches.list(MatchType.VERBATIM)) == 2
    assert match1, match2 in doc_pair_matches.list(MatchType.VERBATIM)


def test_doc_pair_matches_add_same_match_twice():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    doc_pair_matches = DocumentPairMatches(doc1, doc2)
    doc_pair_matches.add(match)
    doc_pair_matches.add(match)
    assert len(doc_pair_matches.list(MatchType.VERBATIM)) == 1
    assert match in doc_pair_matches.list(MatchType.VERBATIM)


def test_doc_pair_matches_add_match_from_other_pair_fails():
    doc1, doc2, doc3 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', ''), \
                       Document('doc3', 'path/to/doc3', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc2), Fragment(6, 9, doc3))
    doc_pair_matches = DocumentPairMatches(doc1, doc2)
    doc_pair_matches.add(match1)
    with pytest.raises(DifferentDocumentPairError):
        doc_pair_matches.add(match2)
    assert len(doc_pair_matches.list(MatchType.VERBATIM)) == 1
    assert doc_pair_matches.doc1, doc_pair_matches.doc2 == (doc1, doc2)


def test_doc_pair_matches_update():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    matches = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2)), \
              Match(MatchType.VERBATIM, Fragment(10, 15, doc2), Fragment(6, 9, doc1))
    doc_pair_matches = DocumentPairMatches(doc1, doc2)
    doc_pair_matches.update(matches)
    assert len(doc_pair_matches.list(MatchType.VERBATIM)) == 2


def test_doc_pair_matches_list():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc1), Fragment(6, 9, doc2))
    doc_pair_matches = DocumentPairMatches(doc1, doc2, {match1, match2})
    matches = doc_pair_matches.list(MatchType.VERBATIM)
    assert matches == {match1, match2}


def test_doc_pair_matches_len():
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    match1 = Match(MatchType.VERBATIM, Fragment(0, 7, doc1), Fragment(0, 4, doc2))
    match2 = Match(MatchType.VERBATIM, Fragment(10, 15, doc1), Fragment(6, 9, doc2))
    doc_pair_matches = DocumentPairMatches(doc1, doc2, {match1, match2})
    assert len(doc_pair_matches) == 2


def _create_sent(doc_name: str, sent_idx: int):
    doc = Document(doc_name, 'path/to/doc', '')
    [doc.sents(include_common=True).add(Sentence(idx, -1, Counter(), doc)) for idx in range(sent_idx + 1)]
    return doc.sents(include_common=True)[sent_idx]
