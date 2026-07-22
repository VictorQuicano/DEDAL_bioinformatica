import numpy as np

from dedal.models.initializers import BLOSUM_62

FALLBACK = {
    'U': 'C',
    'O': 'K',
    'J': 'L',
}


def load_blosum62():
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
    codes = []
    for residue in seq.upper():
        residue = FALLBACK.get(residue, residue)
        codes.append(_INDEX.get(residue, _INDEX['X']))
    return np.asarray(codes, dtype=np.int32)


def score_matrix_for(seq_x, seq_y):
    codes_x = encode(seq_x)
    codes_y = encode(seq_y)
    return _MATRIX[np.ix_(codes_x, codes_y)].astype(np.float64)
