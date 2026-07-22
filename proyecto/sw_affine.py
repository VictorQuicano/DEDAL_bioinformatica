import numpy as np

_MATCH, _GAP_X, _GAP_Y = 0, 1, 2
_START = -1

_NEG_INF = -np.inf


def smith_waterman_affine(seq_x, seq_y, subst,
                          gap_open=11.0, gap_extend=1.0):
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
            open_x = m[i - 1, j] - gap_open
            extend_x = ix[i - 1, j] - gap_extend
            if open_x >= extend_x:
                ix[i, j], ptr_ix[i, j] = open_x, _MATCH
            else:
                ix[i, j], ptr_ix[i, j] = extend_x, _GAP_X

            open_y = m[i, j - 1] - gap_open
            extend_y = iy[i, j - 1] - gap_extend
            if open_y >= extend_y:
                iy[i, j], ptr_iy[i, j] = open_y, _MATCH
            else:
                iy[i, j], ptr_iy[i, j] = extend_y, _GAP_Y

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
                m[i, j], ptr_m[i, j] = 0.0, _START

            if m[i, j] > best_score:
                best_score, best_i, best_j = m[i, j], i, j

    matches = _traceback(m, ptr_m, ptr_ix, ptr_iy, best_i, best_j)
    return best_score, matches


def _traceback(m, ptr_m, ptr_ix, ptr_iy, i, j):
    matches = []
    state = _MATCH

    while i > 0 and j > 0:
        if state == _MATCH:
            if m[i, j] <= 0:
                break
            matches.append((i - 1, j - 1))
            state = ptr_m[i, j]
            i, j = i - 1, j - 1
        elif state == _GAP_X:
            state = ptr_ix[i, j]
            i -= 1
        else:
            state = ptr_iy[i, j]
            j -= 1

    matches.reverse()
    return matches
