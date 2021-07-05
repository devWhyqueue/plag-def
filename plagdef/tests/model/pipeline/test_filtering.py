# The notion of an overlap:
# While Sanchez-Perez et al. define two clusters as overlapping when they share at least one sentence in the
# suspicious document, PlagDef follows a different approach.
# This is because PlagDef takes a set of documents without knowing which are sources and which are plagiarism cases.
# Therefore it can very well be the case that doc1 used a fragment of doc2 multiple times but also vice versa:
# 1) doc1 copied from doc2:
#   A fragment from doc2 is matched multiple times to different fragments in doc1 and therefore belongs to different
#   clusters/cases.
#   But NEVER vice versa for the same fragment.
# 2) doc2 copied from doc1:
#   A fragment from doc1 is matched multiple times to different fragments in doc2 and therefore belongs to different
#   clusters/cases.
#   But NEVER vice versa for the same fragment.
# => So if there is a fragment in doc2 which is part of more than one cluster/case ("doc1 copied from doc2") and these
#   particular clusters simultaneously share a fragment in doc1 ("doc2 copied from doc1"), they are considered
#   overlapping.
# In short PlagDef defines two clusters as overlapping if they share a sentence in doc1's and also in doc2's fragment.
# It then considers both variants 1 and 2 and picks the one of higher quality for each of the overlapping clusters.
# Finally all clusters but the in this sense best cluster are discarded.
from unittest.mock import patch

from networkx import to_dict_of_lists

from plagdef.model.models import Cluster, RatedCluster
from plagdef.model.pipeline.filtering import ClusterFilter, _build_overlap_graph, _resolve_overlaps, \
    _next_overlapping_cluster
from plagdef.tests.model.pipeline.test_extension import _create_seeds


def test_build_overlap_graph():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    cluster_c = Cluster(_create_seeds([(9, 7), (11, 9), (13, 11)]))
    non_ol_cluster_d = Cluster(_create_seeds([(20, 20)]))
    overlap_graph = _build_overlap_graph({cluster_a, cluster_b, cluster_c, non_ol_cluster_d})
    assert to_dict_of_lists(overlap_graph) == {cluster_a: [cluster_b], cluster_b: [cluster_a, cluster_c],
                                               cluster_c: [cluster_b], non_ol_cluster_d: []} \
           or to_dict_of_lists(overlap_graph) == {cluster_a: [cluster_b], cluster_b: [cluster_c, cluster_a],
                                                  cluster_c: [cluster_b], non_ol_cluster_d: []}


def test_build_overlap_graph_with_one_cluster():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    overlap_graph = _build_overlap_graph({cluster_a})
    assert to_dict_of_lists(overlap_graph) == {cluster_a: []}


def test_build_overlap_graph_discards_clusters_only_overlapping_in_one_doc():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    cluster_c = Cluster(_create_seeds([(0, 7)]))
    cluster_d = Cluster(_create_seeds([(14, 4)]))
    cluster_e = Cluster(_create_seeds([(5, 12)]))
    non_ol_cluster_f = Cluster(_create_seeds([(20, 20)]))
    overlap_graph = _build_overlap_graph({cluster_a, cluster_b, cluster_c, cluster_d, cluster_e, non_ol_cluster_f})
    assert to_dict_of_lists(overlap_graph) == {cluster_a: [cluster_b], cluster_b: [cluster_a], cluster_c: [],
                                               cluster_d: [], cluster_e: [], non_ol_cluster_f: []}


def test_resolve_overlaps():
    # Given an adjacent_sents_gap = 1, these overlapping clusters may exist
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    cluster_c = Cluster(_create_seeds([(9, 7), (11, 9), (13, 11)]))
    # common_cluster_ol = {cluster_a: {cluster_b}, cluster_b: {cluster_a, cluster_c}, cluster_c: {cluster_b}}
    with patch.object(Cluster, 'best_with_respect_to', _best_with_respect_to_fake):
        resolved_clusters = _resolve_overlaps({cluster_a, cluster_b, cluster_c})
    assert resolved_clusters == {cluster_a, cluster_c}


def test_remove_small_clusters():
    # char_lengths: doc1: 10, doc2: 11
    cluster_a = Cluster(_create_seeds([(0, 4), (5, 9), (9, 14)]))
    # char_lengths: doc1: 5, doc2: 10
    cluster_b = Cluster(_create_seeds([(9, 7), (11, 12), (13, 16)]))
    # char_lengths: doc1: 11, doc2: 3
    cluster_c = Cluster(_create_seeds([(10, 2), (15, 3), (20, 4)]))
    # char_lengths: doc1: 1, doc2: 1
    cluster_d = Cluster(_create_seeds([(20, 20)]))
    cluster_filter = ClusterFilter(10)
    filtered_clusters = cluster_filter._remove_small_clusters({cluster_a, cluster_b, cluster_c, cluster_d})
    assert filtered_clusters == {cluster_a}


def test_next_overlapping_cluster_with_empty_graph():
    cluster = Cluster(_create_seeds([(0, 4), (5, 9), (9, 14)]))
    graph = _build_overlap_graph({cluster})
    assert _next_overlapping_cluster(graph) is None


def test_next_overlapping_cluster_with_biconnected_graph():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    graph = _build_overlap_graph({cluster_a, cluster_b})
    next_ol_cluster = _next_overlapping_cluster(graph)
    assert next_ol_cluster == cluster_a or next_ol_cluster == cluster_b


def test_next_overlapping_cluster_with_not_biconnected_graph():
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    cluster_c = Cluster(_create_seeds([(9, 7), (11, 9), (13, 11)]))
    graph = _build_overlap_graph({cluster_a, cluster_b, cluster_c})
    assert _next_overlapping_cluster(graph) == cluster_b


def _best_with_respect_to_fake(self: Cluster, ol_cluster: Cluster):
    cluster_a = Cluster(_create_seeds([(0, 4), (3, 2), (6, 0)]))
    cluster_b = Cluster(_create_seeds([(4, 8), (7, 5), (10, 2)]))
    cluster_c = Cluster(_create_seeds([(9, 7), (11, 9), (13, 11)]))
    rated_clusters = {cluster_a: RatedCluster(cluster_a, 0.2, 5),
                      cluster_b: RatedCluster(cluster_b, 0.1, 5),
                      cluster_c: RatedCluster(cluster_c, 0, 5)}
    self_rated = rated_clusters[self]
    ol_cluster_rated = rated_clusters[ol_cluster]
    return self_rated if self_rated >= ol_cluster_rated else ol_cluster_rated
