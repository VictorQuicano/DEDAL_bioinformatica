# coding=utf-8
"""Sample evaluation pairs from Pfam seed alignments, stratified by identity.

Pairs are drawn per identity band rather than uniformly: a uniform draw over
Pfam seeds lands almost entirely above 15% identity, leaving the low-identity
band — the one the comparison is actually about — with a handful of pairs and
a meaningless error bar.
"""

import dataclasses
import itertools
import random

from dedal.preprocessing.alignment import (alignment_from_gapped_sequences,
                                           pid_from_matches)
from proyecto import pfam_data

# Identity bands, matching the stratification used in the paper's Figure 2.
PID_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.01)]

# DEDAL appends an EOS token before padding to 512, so the usable length is
# one less. Leaving a small margin here; longer sequences are truncated
# silently by the model, which would look like a mysterious accuracy drop.
MAX_LEN = 510

# Below this many aligned columns, F1 moves in coarse jumps and a single
# traceback tie-break can swing a whole band.
MIN_MATCHES = 20

# Near-duplicate rows inflate both methods equally and carry no information.
MAX_PID = 0.9


@dataclasses.dataclass
class Pair:
    family: str
    name_x: str
    name_y: str
    seq_x: str
    seq_y: str
    ref: frozenset  # 0-based (i, j) match pairs from the reference alignment
    pid: float


def bin_index(pid):
    """Return the index of the identity band `pid` falls into."""
    for index, (low, high) in enumerate(PID_BINS):
        if low <= pid < high:
            return index
    return None


def candidates_from_family(acc, max_combinations=8000):
    """Yield every usable pair from one family's seed alignment."""
    rows = pfam_data.load_family(acc)
    names = list(rows)

    for name_x, name_y in itertools.islice(
            itertools.combinations(names, 2), max_combinations):
        gapped_x, gapped_y = rows[name_x], rows[name_y]
        matches, (start_x, start_y) = alignment_from_gapped_sequences(
            gapped_x, gapped_y)
        if len(matches) < MIN_MATCHES:
            continue

        seq_x = pfam_data.ungapped(gapped_x)
        seq_y = pfam_data.ungapped(gapped_y)
        if len(seq_x) > MAX_LEN or len(seq_y) > MAX_LEN:
            continue

        pid = pid_from_matches(seq_x, seq_y, matches, start_x, start_y)
        if pid > MAX_PID:
            continue

        ref = frozenset((start_x - 1 + a, start_y - 1 + b) for a, b in matches)
        yield Pair(acc, name_x, name_y, seq_x, seq_y, ref, pid)


def build_dataset(accessions=None, per_bin=50, per_family_cap=20, seed=0):
    """Draw a dataset with a fixed quota of pairs per identity band."""
    accessions = accessions or pfam_data.FAMILIES
    rng = random.Random(seed)

    pools = {index: [] for index in range(len(PID_BINS))}
    for acc in accessions:
        by_bin = {index: [] for index in range(len(PID_BINS))}
        for pair in candidates_from_family(acc):
            index = bin_index(pair.pid)
            if index is not None:
                by_bin[index].append(pair)
        # Cap each family's contribution per band so that a family with
        # hundreds of seed rows cannot dominate a band on its own.
        for index, pairs in by_bin.items():
            rng.shuffle(pairs)
            pools[index].extend(pairs[:per_family_cap])

    dataset = []
    for index, pairs in pools.items():
        rng.shuffle(pairs)
        dataset.append((index, pairs[:per_bin]))
    return dataset
