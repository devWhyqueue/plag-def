from __future__ import annotations

from plagdef.model.models import Seed, Cluster


class ClusterBuilder:
    def __init__(self, adjacent_sents_gap: int, min_adjacent_sents_gap: int, min_sent_number: int,
                 min_cluster_cos_sim: float):
        self._adjacent_sents_gap = adjacent_sents_gap
        self._min_adjacent_sents_gap = min_adjacent_sents_gap
        self._min_sent_number = min_sent_number
        self._min_cluster_cos_sim = min_cluster_cos_sim

    def extend(self, seeds: set[Seed], adjacent_sents_gap: int = None) -> set[Cluster]:
        if adjacent_sents_gap is None:
            adjacent_sents_gap = self._adjacent_sents_gap
        clusters = _build_clusters(seeds, adjacent_sents_gap)
        return self._validate(clusters, adjacent_sents_gap)

    def _validate(self, clusters: set[Cluster], adjacent_sents_gap: int) -> set[Cluster]:
        valid_clusters = set()
        for cluster in clusters:
            if cluster.cos_sim > self._min_cluster_cos_sim:
                valid_clusters.add(cluster)
            elif adjacent_sents_gap > self._min_adjacent_sents_gap:
                cluster_detections = self.extend(set(cluster.seeds), adjacent_sents_gap - 1)
                valid_clusters.update(cluster_detections)
        return valid_clusters


def _build_clusters(seeds: set[Seed], adjacent_sents_gap: int) -> set[Cluster]:
    doc1_clusters = _join_seeds(seeds, adjacent_sents_gap, first=True)
    clusters = set()
    for cluster in doc1_clusters:
        clusters.update(_join_seeds(cluster.seeds, adjacent_sents_gap, first=False))
    return clusters


def _join_seeds(seeds: set[Seed], adjacent_sents_gap: int, first: bool) -> set[Cluster]:
    sorted_seeds = sorted(seeds, key=lambda s: s.sent1.start_char if first else s.sent2.start_char)
    clusters = set()
    seed_iter: enumerate = enumerate(sorted_seeds)
    for seed_idx, seed in seed_iter:
        cluster_seeds = [seed]  # Cluster contains at least first seed
        for adj_seed in sorted_seeds[seed_idx + 1:]:  # For following seeds
            sent1 = cluster_seeds[-1].sent1 if first else cluster_seeds[-1].sent2
            sent2 = adj_seed.sent1 if first else adj_seed.sent2
            if sent1.adjacent_to(sent2, adjacent_sents_gap):  # If sent is adjacent, add seed
                cluster_seeds.append(next(seed_iter)[1])
            else:
                break  # Seeds are sorted by sent start_char
        clusters.add(Cluster(set(cluster_seeds)))
    return clusters
