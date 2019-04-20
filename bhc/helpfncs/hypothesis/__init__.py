import numpy as np
from scipy import stats
import scipy.linalg as la
from scipy.special import gamma, loggamma

### hypothesis evaluation helper functions ###

def eval_H1(ci, cj, **hkparams):
    """
    Evaluate hypothesis 1 for merging individual points.
    This function assumes MVTNormal-Wishart conjugacy.
    Will need to generalize later
    ci and cj - clusters being checked for merge
    
    Inv Wish prior hyperparams:
    r - scaling factor
    v - df 
    
    Default Normal prior hyperparams:
    m - prior mean (currently defined as sample mean)
    S - prior covariance (currently defined as empirical cov mat)
    
    **hkparams kwargs provides mvtnorm and Wishart prior hyperparams
    """

    # keyword arguments assignment
    X = np.vstack([ci.clust, cj.clust])
    N, k = X.shape
    m = hkparams["diffusePriorNorm"]["loc"]
    S = hkparams["diffusePriorNorm"]["scale"]
    v = hkparams["diffusePriorWish"]["df"]
    r = hkparams["diffusePriorWish"]["scale"]
    
    ## version of eval_H1 from Heller appendix
    # posterior precision matrix
    Xrowsum = np.sum(X, axis = 0) # calc once for efficiency
    
    S_prime = (
        S + np.dot(X.T, X) + (r * N * np.dot(m, m.T)) / (N + r) + 
        (1 / (N + r)) * np.dot(Xrowsum, Xrowsum.T) +
        (r / (N + r)) * (np.dot(m, Xrowsum.T) + np.dot(Xrowsum, m.T))
    )

    v_prime = v + N

    # Jonathan - I rearanged this a bit so I could see what was going on piece by piece
    # feel free to modify. Still pretty rough to look at... helper fcns?
    # write out components of log(p(D_k | H_1)) - broken up since it's so long   
    log_pr_D_H1 = (
        np.log(
            (2 * np.pi)**(-N * k / 2) * 
            (
                r / (N + r)**(k / 2) *
                la.det(S)**(v / 2) * 
                la.det(S_prime)**(-v_prime / 2)
                )) + # end of first log
        np.log(2**(v_prime * k / 2)) +
        np.sum(loggamma((np.repeat(v_prime + 1, k) - np.arange(1, k)) / 2)) -
        (
        np.log(2**(v * k / 2)) +
        np.sum(loggamma((np.repeat(v + 1, k) - np.arange(1, k)) / 2))
        )
    )
            
    return np.exp(log_pr_D_H1)

###############################################################################


def eval_H2(ci, cj):
    """
    probability of hypothis 2: data comes from diff distr
    ci and cj - clusters being checked for merge
    """
    return ci.margLik * cj.margLik
    
###############################################################################

def marginal_clust_k(ci, cj, pik, pH1):
    """
    calculate the marginal probability of cluster k existing (marginal
    likelihood) and updated prior for cluster k (ck.pi)
    these objects are returned as a tuple
    This - the 'evidence' in Bayes Rule; called P(D_k | T_k) in paper
    ci and cj - clusters being checked for merge
    ck - cluster formed by ci, cj merge
    pH1 - the probability of merge hypothesis
    """
    marginalLikelihood = pik * pH1 + (1 - pik) * eval_H2(ci, cj)
    return marginalLikelihood


###############################################################################

def posterior_join_k(ci, cj, hkparams):
    """
    calculate posterior probability of merged trees i & j
    ci and cj - clusters being checked for merge
    hkparams - dictionary of distribution parameters
    """

    # cluster size from proposed cluster k
    propClustSize = ci.clustsize + cj.clustsize
    
    # d_k for proposed cluster k
    propd = (
        	hkparams["clusterConcentrationParam"]["alpha"] *
            gamma(propClustSize) +
        	ci.d * cj.d
        	)
        
    # prior for creating proposed cluster k
    pik = ci.alpha * gamma(propClustSize) / propd

    # hypothesis 1 for cluster i and cluster j
    ijH1 = eval_H1(ci, cj, **hkparams)
    # marginal likelihood for cluster k; hypothesis 2 calculated within
    ijMarg = marginal_clust_k(ci, cj, pik, ijH1)
    # posterior probability for creating cluster k from ci and cj
    rk = pik * ijH1 / ijMarg
    return rk












