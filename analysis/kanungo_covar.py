# Calculate the covariance from pattern histograms to use for
# Mahalanobis distance which is minimized in Kanungo estimation
import numpy as np

hists = np.loadtxt('results/kanungo_3x3_hist.csv.gz', delimiter=',')
# Remove all-white pattern and normalize other patterns
hists = hists[:, 1:].astype(float)
hists /= hists.sum(1)[:, None]
V = np.cov(hists.T)
diag = V[range(511), range(511)]
V[range(511), range(511)] += np.mean(diag) / 100
VI = np.linalg.inv(V)

np.savez('results/kanungo_covar_inv.npz', VI=VI)
