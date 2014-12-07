import numpy as np

def match(x, y):
    """ For each x, return the index of the closest value in y """
    dist = np.abs(x[:, None] - y[None, :])
    return np.argmin(dist, axis=1)

def merge(x, y, thresh):
    """ Merge each value in x with a value in y if they are within thresh.
        Return array of shape (len_pairs, 2) with index of the element of
        the pair in x and y.
        If there is not a corresponding value in one array, the index is -1. """
    pairdist = np.abs(x[:, None] - y[None, :])
    x_closest = np.argmin(pairdist, axis=1)
    dist = pairdist[np.arange(len(x)), x_closest]
    x_ispair = dist <= thresh
    x_ind = np.arange(len(x))
    y_ind = np.arange(len(y))
    x_pair = np.c_[x_ind[x_ispair], x_closest[x_ispair]]
    x_nopair = np.c_[x_ind[~x_ispair], np.repeat(-1, (~x_ispair).sum())]
    y_ispair = np.in1d(np.arange(len(y)), x_pair[:,1])
    y_nopair = np.c_[np.repeat(-1, (~y_ispair).sum()), y_ind[~y_ispair]]
    pairs = np.r_[x_pair, x_nopair, y_nopair]
    order = np.argsort(pairs.mean(axis=1))
    return pairs[order]

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
