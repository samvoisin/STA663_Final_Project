import numpy as np
from scipy import stats
import scipy.linalg as la
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
    r - scaling factor
    
    Default Normal prior hyperparams:
    m - prior mean (currently defined as sample mean)
    S - prior covariance (currently defined as empirical cov mat)
    
    hkparams - dictionary of prior parameters (e.g.
    diffuseWishPrior, diffuseNormPrior, clusterConcentrationPrior)
    """

    # abbreviated keyword arguments for brevity in calcs;
    # not the most efficient thing to do, but it's here for now...
    X = np.vstack([ci.clust, cj.clust])
    N, k = X.shape
    m = ci.priorParams["diffuseNormPrior"]["loc"]
    S = ci.priorParams["diffuseNormPrior"]["scale"]
    v = ci.priorParams["diffuseWishPrior"]["df"]
    r = ci.priorParams["diffuseWishPrior"]["scale"]
    
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
            (r / (N + r)**(k / 2) *
            la.det(S)**(v / 2) * 
            la.det(S_prime)**(-v_prime / 2))
            ) + # end of first log
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












