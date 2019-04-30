import numpy as np
import scipy.linalg as la
from scipy import stats
from functools import reduce
from scipy.special import gamma, loggamma

from bhc.helpfncs.distrs.normal_invwish import norm_inv_wish_H1


### hypothesis evaluation helper functions ###

def eval_H1(ci, cj):
    """
    Evaluate hypothesis 1 for merging two clusters ci and cj
    based on the family specified when HierarchyTree is
    initialized.

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

    if ci.family == "norm-invwish":
        marginalLikelihood = norm_inv_wish_H1(ci, cj)
    elif ci.family == "beta-bern":
        pass

    return marginalLikelihood

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

    # gamma fcn for the proposed cluster size 
    # calculating once for efficiency
    gammaPCS = gamma(propClustSize)

    # d_k for proposed cluster k
    if propClustSize == 2: # when joining two singletons
        propd = ci.alpha
    else:
        propd = (
        	ci.alpha *
            gammaPCS +
        	ci.d * cj.d
        	)

    # probability of creating proposed cluster k
    pik = ci.alpha * gammaPCS / propd

    return pik


###############################################################################


def marginal_clust_k(ci, cj):
    """
    calculate the marginal likelihood for cluster k
    This is the 'evidence' in Bayes Rule; called P(D_k | T_k) in paper
    ci and cj - clusters being tested for merge

    This function also returns the probability of the merge
    hypothesis - pH1; this is done for efficiency
    """
    
    # calculate probability of creating proposed cluster k
    pik = prop_pi_k(ci, cj)

    # calculate probability of merge hypothesis
    pH1 = eval_H1(ci, cj)

    # calculate probability of independent hypothesis
    pH2 = eval_H2(ci, cj)

    # calculate marginal likelihood of cluster k
    marginalLikelihood = pik * pH1 + (1 - pik) * pH2

    return pH1, marginalLikelihood


###############################################################################


def posterior_join_k(ci, cj):
    """
    calculate posterior probability of merged trees i & j
    ci and cj - clusters being tested for merge
    """

    # marginal likelihood for cluster k; hypothesis 2 calculated within
    ijH1, ijMarg = marginal_clust_k(ci, cj)

    # calculate probability of creating proposed cluster k
    pik = prop_pi_k(ci, cj)

    # posterior probability for creating cluster k from ci and cj
    tol = 1e-60 # if marginal probability approaches zero
    if ijMarg < tol:
        rk = 0.0 # set rk == 0 to prevent num errors. will be pruned. 
    else:
        rk = pik * ijH1 / ijMarg

    return rk
