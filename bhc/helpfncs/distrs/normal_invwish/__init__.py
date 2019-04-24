import numpy as np
import scipy.linalg as la
from scipy import stats
from functools import reduce
from scipy.special import gamma, loggamma


### hypothesis evaluation helper functions ###

def norm_inv_wish_H1(ci, cj):
    """
    Evaluate hypothesis 1 for merging two clusters ci and cj in
	the mutltivariate normal-Wishart conjugacy case.

	ci and cj - clusters being tested for merge. These objects
	contain prior hyperparameters to be used.

    Inv Wish prior hyperparams:
    v - df
    S - prior precision matrix (currently defined as empirical cov mat)

    Default Normal prior hyperparams:
    m - prior mean (currently defined as sample mean)
    r - scaling factor on the prior precision of the mean
    """

    # abbreviated keyword arguments for brevity in calcs;
    # not the most efficient thing to do, but it's here for now...
    X = np.vstack([ci.clust, cj.clust])
    N, k = X.shape
    m = ci.priorParams["diffuseNormPrior"]["loc"]
    r = ci.priorParams["diffuseNormPrior"]["meanscale"]
    v = ci.priorParams["diffuseInvWishPrior"]["df"]
    S = ci.priorParams["diffuseInvWishPrior"]["scale"]

    ## version of eval_H1 from Heller appendix
    # posterior precision matrix
    Xrowsum = np.sum(X, axis = 0) # calc once for efficiency

    Sprime = (
        S + np.dot(X.T, X) + (r * N / (N + r)) * np.dot(m, m.T) +
        (1 / (N + r)) * np.dot(Xrowsum, Xrowsum.T) +
        (r / (N + r)) * (np.dot(m, Xrowsum.T) + np.dot(Xrowsum, m.T))
    )

    vprime = v + N

    # components of p(D_k | H_1)
    numer = [gamma((vprime + 1 - d) / 2) for d in range(1, k + 1)]
    numer = reduce(lambda x, y: x * y, numer)
    denom = [gamma((v + 1 - d) / 2) for d in range(1, k + 1)]
    denom = reduce(lambda x, y: x * y, denom)

    fact = (2 ** (vprime*k/2) / 2 ** (v*k/2)) * (numer / denom)

    MarginalLikelihood = (
        (-N * k / 2) * np.log(2 * np.pi) +
        (k / 2) * np.log(r / (N + r)) +
        (v / 2) * np.log(la.det(S)) +
        (-vprime / 2) * np.log(la.det(Sprime)) +
        np.log(fact)
    )

    return np.exp(MarginalLikelihood)
