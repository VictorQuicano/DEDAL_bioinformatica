import numpy as np

from dedal import infer


def prf1(pred, ref):
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
    inputs = infer.preprocess(seq_x, seq_y)
    output = model(inputs)
    _, states, _ = infer.postprocess(
        [output['sw_scores'], output['paths'], output['sw_params']],
        len(seq_x), len(seq_y))
    return set(map(tuple, np.argwhere(states.numpy()[:, :, 0] > 0)))
