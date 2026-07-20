# coding=utf-8
"""BLOSUM62 substitution matrix, parsed from DEDAL's embedded constant.

The matrix ships as a string literal inside the paper's own codebase, so the
baseline and the model are scored against the exact same table.
"""

import numpy as np

from dedal.models.initializers import BLOSUM_62

# Residues Pfam seeds contain that BLOSUM62's 24-letter alphabet does not.
# Mapping them to their closest standard residue avoids a KeyError mid-run.
FALLBACK = {
    'U': 'C',  # selenocysteine  -> cysteine
    'O': 'K',  # pyrrolysine     -> lysine
    'J': 'L',  # Leu/Ile ambiguous
}


def load_blosum62():
    """Parse BLOSUM_62 into (letter -> index, 24x24 int matrix)."""
    lines = [ln for ln in BLOSUM_62.strip().splitlines() if ln.strip()]
    header = lines[0].split()
    index = {letter: i for i, letter in enumerate(header)}

    matrix = np.zeros((len(header), len(header)), dtype=np.int32)
    for row_num, line in enumerate(lines[1:]):
        fields = line.split()
        if fields[0] != header[row_num]:
            raise ValueError(
                f'Row {row_num} is labelled {fields[0]}, expected {header[row_num]}')
        matrix[row_num] = [int(v) for v in fields[1:]]

    if not np.array_equal(matrix, matrix.T):
        raise ValueError('BLOSUM62 must be symmetric')
    return index, matrix


_INDEX, _MATRIX = load_blosum62()


def encode(seq):
    """Map a protein sequence to BLOSUM62 row indices."""
    codes = []
    for residue in seq.upper():
        residue = FALLBACK.get(residue, residue)
        # Anything still unrecognised falls back to 'X' (any residue).
        codes.append(_INDEX.get(residue, _INDEX['X']))
    return np.asarray(codes, dtype=np.int32)


def score_matrix_for(seq_x, seq_y):
    """Return the len_x x len_y matrix of pairwise substitution scores."""
    codes_x = encode(seq_x)
    codes_y = encode(seq_y)
    return _MATRIX[np.ix_(codes_x, codes_y)].astype(np.float64)
