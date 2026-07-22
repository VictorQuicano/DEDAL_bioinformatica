import dataclasses
import itertools
import random

from dedal.preprocessing.alignment import (alignment_from_gapped_sequences,
                                           pid_from_matches)
from proyecto import pfam_data

PID_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 1.01)]

MAX_LEN = 510

MIN_MATCHES = 20

MAX_PID = 0.9


@dataclasses.dataclass
class Pair:
    family: str
    name_x: str
    name_y: str
    seq_x: str
    seq_y: str
    ref: frozenset
    pid: float


def bin_index(pid):
    for index, (low, high) in enumerate(PID_BINS):
        if low <= pid < high:
            return index
    return None


def candidates_from_family(acc, max_combinations=8000):
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
    accessions = accessions or pfam_data.FAMILIES
    rng = random.Random(seed)

    pools = {index: [] for index in range(len(PID_BINS))}
    for acc in accessions:
        by_bin = {index: [] for index in range(len(PID_BINS))}
        for pair in candidates_from_family(acc):
            index = bin_index(pair.pid)
            if index is not None:
                by_bin[index].append(pair)
        for index, pairs in by_bin.items():
            rng.shuffle(pairs)
            pools[index].extend(pairs[:per_family_cap])

    dataset = []
    for index, pairs in pools.items():
        rng.shuffle(pairs)
        dataset.append((index, pairs[:per_bin]))
    return dataset
