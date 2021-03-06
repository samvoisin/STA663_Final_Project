import numpy as np
import scipy.linalg as la
from scipy import stats
from functools import reduce
from scipy.special import gamma, loggamma


### hypothesis evaluation helper functions ###

def norm_inv_wish_H1(ci, cj):
    """
    Evaluate hypothesis 1 for merging two clusters ci and cj.
    This function assumes MVTNormal-Wishart conjugacy.
    Will need to generalize later
    ci and cj - clusters being tested for merge
    
    Inv Wish prior hyperparams:
    v - df
    S - prior precision matrix (currently defined as empirical cov mat)
    
    Default Normal prior hyperparams:
    m - prior mean (currently defined as sample mean)
    r - scaling factor on the prior precision of the mean
    
    hkparams - dictionary of prior parameters (e.g.
    diffuseWishPrior, diffuseNormPrior, clusterConcentrationPrior)
    """

    # abbreviated keyword arguments for brevity in calcs
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
        np.dot(X.T, X) + (r * N / (N + r)) * np.dot(m, m.T) + 
        (1 / (N + r)) * np.dot(Xrowsum, Xrowsum.T) -
        (r / (N + r)) * (np.dot(m, Xrowsum.T) + np.dot(Xrowsum, m.T)) + S
    )

    vprime = v + N

    # components of p(D_k | H_1)
    numer = [loggamma((vprime + 1 - d) / 2) for d in range(1, k + 1)]
    #numer = reduce(lambda x, y: x * y, numer)
    numer = np.sum(numer)
    denom = [loggamma((v + 1 - d) / 2) for d in range(1, k + 1)]
    #denom = reduce(lambda x, y: x * y, denom)
    denom = np.sum(denom)
    
    logfact = (vprime*k/2)*np.log(2) + numer - (v*k/2)*np.log(2) - denom
    
    
    MarginalLikelihood = (
        (-N * k / 2) * np.log(2 * np.pi) +
        (k / 2) * np.log(r / (N + r)) +
        (v / 2) * np.log(la.det(S)) +
        (-vprime / 2) * np.log(la.det(Sprime)) +
        logfact
    )
    
    return np.exp(MarginalLikelihood)
