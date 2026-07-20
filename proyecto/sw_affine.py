# coding=utf-8
"""Smith-Waterman local alignment with affine gap penalties (Gotoh).

This is the algorithm from the course, extended along two axes so it can be
compared against DEDAL on protein sequences:

  * lab_02 implemented local SW with a linear gap and a match/mismatch score
    for DNA. Here the score comes from a substitution matrix (BLOSUM62).
  * lab_04 implemented an affine gap (Gotoh) but for *global* alignment
    (Needleman-Wunsch). Here the affine gap is combined with local alignment:
    the zero option in the recurrence, and traceback stopping at zero.

Three matrices are needed for an affine gap, because the cost of a gap depends
on whether it is the first gapped position or a continuation:

  M[i, j]   best score for an alignment ending with x_i aligned to y_j
  Ix[i, j]  best score for an alignment ending with x_i aligned to a gap
  Iy[i, j]  best score for an alignment ending with y_j aligned to a gap

A gap of length k costs `gap_open + (k - 1) * gap_extend`.
"""

import numpy as np

# Traceback state codes.
_MATCH, _GAP_X, _GAP_Y = 0, 1, 2
_START = -1  # local alignment begins here (the zero option won)

_NEG_INF = -np.inf


def smith_waterman_affine(seq_x, seq_y, subst,
                          gap_open=11.0, gap_extend=1.0):
    """Align two protein sequences locally with affine gaps.

    Args:
      seq_x: protein sequence as a string.
      seq_y: protein sequence as a string.
      subst: `len(seq_x) x len(seq_y)` substitution score matrix.
      gap_open: cost of opening a gap (a positive number; it is subtracted).
      gap_extend: cost of each additional gapped position.

    Returns:
      A `(score, matches)` tuple, where `matches` is the list of 0-based
      `(i, j)` index pairs aligned in a match state, ordered along the
      alignment.
    """
    len_x, len_y = len(seq_x), len(seq_y)
    if subst.shape != (len_x, len_y):
        raise ValueError(
            f'subst has shape {subst.shape}, expected {(len_x, len_y)}')

    m = np.zeros((len_x + 1, len_y + 1))
    ix = np.full((len_x + 1, len_y + 1), _NEG_INF)
    iy = np.full((len_x + 1, len_y + 1), _NEG_INF)

    ptr_m = np.full((len_x + 1, len_y + 1), _START, dtype=np.int8)
    ptr_ix = np.zeros((len_x + 1, len_y + 1), dtype=np.int8)
    ptr_iy = np.zeros((len_x + 1, len_y + 1), dtype=np.int8)

    best_score, best_i, best_j = 0.0, 0, 0

    for i in range(1, len_x + 1):
        for j in range(1, len_y + 1):
            # Gap in y: consume x_i, y stays put.
            open_x = m[i - 1, j] - gap_open
            extend_x = ix[i - 1, j] - gap_extend
            if open_x >= extend_x:
                ix[i, j], ptr_ix[i, j] = open_x, _MATCH
            else:
                ix[i, j], ptr_ix[i, j] = extend_x, _GAP_X

            # Gap in x: consume y_j, x stays put.
            open_y = m[i, j - 1] - gap_open
            extend_y = iy[i, j - 1] - gap_extend
            if open_y >= extend_y:
                iy[i, j], ptr_iy[i, j] = open_y, _MATCH
            else:
                iy[i, j], ptr_iy[i, j] = extend_y, _GAP_Y

            # Match state. Tie-break order is fixed at M > Ix > Iy so that
            # results are reproducible.
            best_prev = m[i - 1, j - 1]
            prev_state = _MATCH
            if ix[i - 1, j - 1] > best_prev:
                best_prev, prev_state = ix[i - 1, j - 1], _GAP_X
            if iy[i - 1, j - 1] > best_prev:
                best_prev, prev_state = iy[i - 1, j - 1], _GAP_Y

            candidate = best_prev + subst[i - 1, j - 1]
            if candidate > 0:
                m[i, j], ptr_m[i, j] = candidate, prev_state
            else:
                # Local alignment: never carry a negative prefix forward.
                m[i, j], ptr_m[i, j] = 0.0, _START

            if m[i, j] > best_score:
                best_score, best_i, best_j = m[i, j], i, j

    matches = _traceback(m, ptr_m, ptr_ix, ptr_iy, best_i, best_j)
    return best_score, matches


def _traceback(m, ptr_m, ptr_ix, ptr_iy, i, j):
    """Walk back from the best cell to where the local alignment started."""
    matches = []
    state = _MATCH

    while i > 0 and j > 0:
        if state == _MATCH:
            # A zero in the match matrix is where this local alignment began.
            if m[i, j] <= 0:
                break
            matches.append((i - 1, j - 1))
            state = ptr_m[i, j]
            i, j = i - 1, j - 1
        elif state == _GAP_X:
            state = ptr_ix[i, j]
            i -= 1
        else:  # _GAP_Y
            state = ptr_iy[i, j]
            j -= 1

    matches.reverse()
    return matches
