from __future__ import annotations

from collections import Counter
from unittest.mock import patch

from plagdef.model.models import Sentence
from plagdef.model.pipeline.extension import ClusterBuilder, Cluster, _build_clusters
from plagdef.model.pipeline.preprocessing import Document
from plagdef.model.pipeline.seeding import Seed


def test_clusters_are_equal():
    cluster1 = Cluster(set(_create_seeds([(0, 0), (3, 1)])))
    cluster2 = Cluster(set(_create_seeds([(3, 1), (0, 0)])))
    assert cluster1 == cluster2


def test_cluster_sets_are_equal():
    cluster1 = Cluster(set(_create_seeds([(0, 0), (3, 1)])))
    cluster2 = Cluster(set(_create_seeds([(3, 1), (0, 0)])))
    assert {cluster1, cluster2} == {cluster2, cluster1}


def test_clustering_with_no_seeds():
    clusters = _build_clusters(set(), 4)
    assert clusters == set()


def test_clustering_with_equal_clusters():
    # The given seeds form these clusters:
    # doc1: [[s0, s1], [s2, s3]]
    # doc2: [[s0, s1], [s2, s3]]
    # (doc1_sent_idx, doc2_sent_idx)
    seeds = _create_seeds([(0, 0), (3, 1), (9, 7), (10, 9)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0], seeds[1]}), Cluster({seeds[2], seeds[3]})}


def test_clustering_with_more_clusters_in_doc1():
    # The given seeds form these clusters:
    # doc1: [[s0, s1], [s2]]
    # doc2: [[s0, s1, s2]]
    # (doc1_sent_idx, doc2_sent_idx)
    seeds = _create_seeds([(0, 0), (3, 1), (9, 6)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0], seeds[1]}), Cluster({seeds[2]})}


def test_clustering_with_more_clusters_in_doc2():
    # The given seeds form these clusters:
    # doc1: [[s0, s1, s2]]
    # doc2: [[s0, s1], [s2]]
    # (doc1_sent_idx, doc2_sent_idx)
    seeds = _create_seeds([(0, 0), (3, 1), (7, 8)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0], seeds[1]}), Cluster({seeds[2]})}


def test_clustering_with_totally_different_clusters():
    # The given seeds form these clusters:
    # doc1: [[s0, s1, s2, s3]]
    # doc2: [[s0], [s1], [s2], [s3]]
    # (doc1_sent_idx, doc2_sent_idx)
    seeds = _create_seeds([(0, 0), (3, 6), (5, 12), (7, 18)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0]}), Cluster({seeds[1]}),
                        Cluster({seeds[2]}), Cluster({seeds[3]})}


def test_clustering_with_seed_which_is_part_of_different_clusters():
    # doc1: [[s0], [s1, s2], [s3], [s4]]
    # doc2: [[s0], [s1], [s2, s3], [s4]]
    seeds = _create_seeds([(18, 17), (36, 33), (37, 49), (54, 53), (74, 63)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0]}), Cluster({seeds[1]}), Cluster({seeds[2]}),
                        Cluster({seeds[3]}), Cluster({seeds[4]})}


def test_clustering_with_three_seeds_in_cluster():
    # doc1: [[s0], [s1, s2, s3], [s4]]
    # doc2: [[s0], [s1], [s2, s3], [s4]]
    seeds = _create_seeds([(2, 17), (21, 34), (25, 43), (30, 45), (50, 51)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0]}), Cluster({seeds[1]}),
                        Cluster({seeds[2], seeds[3]}), Cluster({seeds[4]})}


def test_clustering_processing_order_is_important():
    # doc1: [[s0, s1, s2], [s3, s4, s5]]
    # doc2: [[s0, s1], [s2, s3], [s4, s5]]
    seeds = _create_seeds([(4, 0), (7, 2), (10, 12), (19, 14), (21, 23), (26, 28)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0], seeds[1]}),
                        Cluster({seeds[2]}), Cluster({seeds[3]}),
                        Cluster({seeds[4], seeds[5]})}


def test_clustering_seed_order_makes_no_difference():
    # doc1: [[s0, s1]]
    # doc2: [[s1, s0]]
    seeds = _create_seeds([(0, 4), (5, 1)])
    clusters = _build_clusters(set(seeds), 4)
    assert clusters == {Cluster({seeds[0], seeds[1]})}


def test_validate_with_only_similar_clusters():
    seeds = _create_seeds([(0, 2), (2, 4), (10, 5), (11, 7), (14, 11)])
    extender = ClusterBuilder(4, 0, 1, 0.34)
    # Calculation of cluster similarity untested
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.return_value = 1
        clusters = [Cluster({seeds[0], seeds[1]}),
                    Cluster({seeds[2], seeds[3], seeds[4]})]
        filtered_clusters = extender._validate(set(clusters), 4)
    assert filtered_clusters == {clusters[0], clusters[1]}


def test_validate_with_cluster_with_similarity_below_th():
    seeds = _create_seeds([(0, 2)])
    extender = ClusterBuilder(4, 0, 1, 0.34)
    # Calculation of cluster similarity untested
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        cos_sim.return_value = 0
        clusters = [Cluster({seeds[0]})]
        filtered_clusters = extender._validate(set(clusters), 4)
    assert filtered_clusters == set()


def test_validate_with_cluster_which_is_reduced():
    seeds = _create_seeds([(0, 5), (1, 7), (6, 11)])
    extender = ClusterBuilder(4, 0, 1, 0.34)
    # Calculation of cluster similarity untested
    with patch('plagdef.model.models.util.cos_sim') as cos_sim:
        # idx 0: Create list 'clusters' with one cluster and cos_sim = 0
        # idx 1-3: doc1_clusters: Cluster({seeds[0], seeds[1]}), Cluster({seeds[2]})
        # idx 4-6: Check if doc1_clusters are possible in doc2: Cluster({seeds[0], seeds[1]}), Cluster({seeds[2]})
        # idx 7-8: Create list 'expected_clusters' with two clusters and cos_sim = 1
        cos_sim.side_effect = [0, 1, 1, 1, 1, 1, 1]
        clusters = [Cluster({seeds[0], seeds[1], seeds[2]})]
        filtered_detections = extender._validate(set(clusters), 4)
        expected_clusters = {Cluster({seeds[0], seeds[1]}), Cluster({seeds[2]})}
    assert filtered_detections == expected_clusters


def _create_seeds(seed_tpls: list[tuple]):
    doc1, doc2 = Document('doc1', 'path/to/doc1', ''), Document('doc2', 'path/to/doc2', '')
    max_idx_doc1, max_idx_doc2 = max(seed_tpls, key=lambda seed: seed[0]), \
                                 max(seed_tpls, key=lambda seed: seed[1])
    [doc1.add_sent(Sentence(idx, idx + 1, Counter(), doc1)) for idx in range(max_idx_doc1[0] + 1)]
    [doc2.add_sent(Sentence(idx, idx + 1, Counter(), doc2)) for idx in range(max_idx_doc2[1] + 1)]
    seeds = []
    for seed_tpl in seed_tpls:
        seeds.append(Seed(doc1.sents(include_common=True)[seed_tpl[0]],
                          doc2.sents(include_common=True)[seed_tpl[1]], 1, 1))
    return seeds
