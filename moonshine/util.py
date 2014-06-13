import numpy as np

def match(x, y):
    """ For each x, return the index of the closest value in y """
    dist = np.abs(x[:, None] - y[None, :])
    return np.argmin(dist, axis=1)

# Faster versions of scipy.ndimage functions for 1D arrays
def label_1d(seq):
    seq = seq.astype(bool)
    obj_starts = seq.copy()
    obj_starts[1:] &= ~ seq[:-1]
    obj_nums = np.cumsum(obj_starts)
    labels = obj_nums * seq
    num_labels = obj_nums[-1]
    return labels, num_labels if num_labels > 0 else 1

def center_of_mass_1d(seq):
    """ Returns an ndarray, unlike scipy counterpart """
    seq_ind = np.arange(len(seq))
    seq_sums = np.bincount(seq, weights=seq_ind)[1:]
    seq_counts = np.bincount(seq)[1:]
    return seq_sums / seq_counts
