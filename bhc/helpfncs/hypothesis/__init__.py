import numpy as np
import scipy.linalg as la
from scipy import stats
from functools import reduce
from scipy.special import gamma, loggamma


### hypothesis evaluation helper functions ###

def eval_H1(ci, cj):
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
        (2 * np.pi) ** (-N * k / 2) *
        (r / (N + r)) ** (k / 2) *
        la.det(S) ** (v / 2) *
        la.det(Sprime) ** (vprime / 2) *
        fact
    )
    
    return MarginalLikelihood

###############################################################################


def eval_H2(ci, cj):
    """
    probability of hypothis 2: data comes from diff distr
    ci and cj - clusters being tested for merge
    """
    return ci.margLik * cj.margLik
    
###############################################################################


def prop_pi_k(ci, cj):
    """
    calculate the probability of merging cluster i and cluster j to create
    proposed cluster k
    ci and cj - clusters being tested for merge
    """

    # cluster size from proposed cluster k
    propClustSize = ci.clustsize + cj.clustsize

    # d_k for proposed cluster k
    propd = (
        	ci.alpha *
            gamma(propClustSize) +
        	ci.d * cj.d
        	)

    # probability of creating proposed cluster k
    pik = ci.alpha * gamma(propClustSize) / propd

    return pik


###############################################################################


def marginal_clust_k(ci, cj):
    """
    calculate the marginal likelihood for cluster k
    This is the 'evidence' in Bayes Rule; called P(D_k | T_k) in paper
    ci and cj - clusters being tested for merge
    """
    
    # calculate probability of creating proposed cluster k
    pik = prop_pi_k(ci, cj)

    # calculate probability of merge hypothesis
    pH1 = eval_H1(ci, cj)

    # calculate probability of independent hypothesis
    pH2 = eval_H2(ci, cj)

    # calculate marginal likelihood of cluster k
    marginalLikelihood = pik * pH1 + (1 - pik) * pH2

    return marginalLikelihood


###############################################################################


def posterior_join_k(ci, cj):
    """
    calculate posterior probability of merged trees i & j
    ci and cj - clusters being tested for merge
    """

    # hypothesis 1 for cluster i and cluster j
    ijH1 = eval_H1(ci, cj)

    # marginal likelihood for cluster k; hypothesis 2 calculated within
    ijMarg = marginal_clust_k(ci, cj)

    # calculate probability of creating proposed cluster k
    pik = prop_pi_k(ci, cj)

    # posterior probability for creating cluster k from ci and cj
    rk = pik * ijH1 / ijMarg

    return rk