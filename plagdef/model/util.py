import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import RLock
from typing import Callable

from numpy import dot, array_split
from numpy.linalg import norm
from pkg_resources import get_distribution
from tqdm import tqdm


def version():
    return get_distribution('plagdef').version


def truncate(string: str, length: int):
    return f'{string}' if len(string) < length else f'{string[:length]}...'


def cos_sim(bow1: dict, bow2: dict):
    """
    Compute the cosine similarity cos-sim = sum_i=1_n{a_i * b_i} / sqrt{sum_i=1_n{(a_i)^2}} * sqrt{sum_i=1_n{(
    b_i)^2}}
    """
    aligned_vecs = [(bow1[lemma], bow2[lemma]) for lemma in bow1 if lemma in bow2]
    a, b = [v[0] for v in aligned_vecs], [v[1] for v in aligned_vecs]
    euclidean_norms = norm(list(bow1.values())) * norm(list(bow2.values()))
    return dot(a, b) / euclidean_norms if euclidean_norms else 0


def dice_sim(bow1: dict, bow2: dict):
    """
    Compute the dice similarity dice-coeff = 2 * n_com / (n_x + n_y), n_com being the number of distinct common
    lemmas in both sents, n_x the number of lemmas in the first bow and n_y the number of lemmas in the second bow
    """
    n_com = len(set(bow1.keys()).intersection(bow2.keys()))
    n_x_plus_n_y = len(bow1) + len(bow2)
    return 2 * n_com / n_x_plus_n_y if n_x_plus_n_y else 0


def parallelize(fun: Callable, data):
    data_chunks = array_split(data, os.cpu_count())
    with ProcessPoolExecutor(initargs=(RLock(),), initializer=tqdm.set_lock, max_workers=os.cpu_count()) as p:
        futures = []
        for i, chunk in enumerate(data_chunks):
            futures.append(p.submit(fun, chunk, i))
        match_chunks = [f.result() for f in as_completed(futures)]
    return [match for chunk in match_chunks for match in chunk]
