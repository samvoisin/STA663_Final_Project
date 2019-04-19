import numpy as np
from scipy import stats
from scipy import special as sp

### hypothesis evaluation helper functions ###

def eval_H1(ci, cj, **hkparams):
    """
    Evaluate hypothesis 1 for merging individual points.
    This function assumes MVTNormal-Wishart conjugacy.
    Will need to generalize later
    ci and cj are clusters we are checking to merge
    
    Inv Wish prior hyperparams:
    r - scaling factor
    v - df 
    
    Default Normal prior hyperparams:
    m - prior mean (currently defined as sample mean)
    S - prior covariance (currently defined as empirical cov mat)
    
    **hkparams kwargs provides mvtnorm and Wishart prior hyperparams
    """

    # keyword arguments assignment
    X = np.vstack([clusti.clust, clustj.clust])
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
        np.sum(sp.loggamma((np.repeat(v_prime + 1, k) - np.arange(1, k)) / 2)) -
        (
        np.log(2**(v * k / 2)) +
        np.sum(sp.loggamma((np.repeat(v + 1, k) - np.arange(1, k)) / 2))
        )
    )
            
    return np.exp(log_pr_D_H1)

################################################################################


def eval_H2(ci, cj):
    """
    probability of hypothis 2: data comes from diff distr
    ci and cj are clusters we are checking to merge
    """
    return ci.margLik * cj.margLik
    

def marginal_clust_k(ci, cj, pH1):
    """
    calculate the marginal probability of cluster k existing
    This - the 'evidence' in Bayes Rule; called P(D_k | T_k) in paper
    ci and cj - clusters we are checking to merge
    pH1 - the probability of merge hypothesis
    pH2 - the probability of indep hypothesis
    """
    pik = ck.pi
    marginalLikelihood = ck.pi * pH1 + (1 - ck.pi) * eval_H2(ci, cj)
    return marginalLikelihood














