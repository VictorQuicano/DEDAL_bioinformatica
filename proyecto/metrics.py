# coding=utf-8
"""Alignment quality metrics.

An alignment is represented as the set of 0-based `(i, j)` index pairs it puts
in a match state. Quality is then set agreement against the reference: no
partial credit, so aligning residue `i` to `j + 1` counts as a false positive
*and* a false negative.

DEDAL's own `train/align_metrics.py` computes the same quantity, but as
TensorFlow streaming metrics wired around the training loop's
`(solution_values, solution_paths, sw_params)` tuples. Feeding a NumPy
baseline through that interface would mean fabricating tensors, so this takes
the paper's definition rather than its plumbing.
"""

import numpy as np

from dedal import infer


def prf1(pred, ref):
    """Return (precision, recall, f1) of a predicted match set."""
    if not ref:
        raise ValueError('Reference alignment is empty')
    if not pred:
        return 0.0, 0.0, 0.0

    true_positives = len(pred & ref)
    precision = true_positives / len(pred)
    recall = true_positives / len(ref)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def dedal_matches(model, seq_x, seq_y):
    """Align with DEDAL and return its match set as 0-based (i, j) pairs."""
    inputs = infer.preprocess(seq_x, seq_y)
    output = model(inputs)
    # The TF Hub model returns semantically named keys, so `infer.expand` is a
    # no-op here and `infer.align` would crash on them. Build the tuple
    # directly instead.
    _, states, _ = infer.postprocess(
        [output['sw_scores'], output['paths'], output['sw_params']],
        len(seq_x), len(seq_y))
    # State index 0 is 'match' (1 and 2 are gap_open and gap_extend).
    return set(map(tuple, np.argwhere(states.numpy()[:, :, 0] > 0)))
