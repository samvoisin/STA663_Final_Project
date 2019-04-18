import numpy as np
from scipy import stats

### hypothesis evaluation helper functions ###

def eval_H1(ci, cj, **hkparams):
    """probability of hypothes 1: data comes from same distribution
    ci and cj are clusters we are checking to merge into cluster k
    ss is the sample size for rejection sampler
    **hkparams kwargs is prior parameters for cluster"""
    
    # prior distributions
    meanPriors = stats.norm.rvs(**hkparams["diffusePriorNorm"], size = 100)
    precisPriors = stats.gamma.rvs(**hkparams["diffusePriorGamma"], size = 100)
    
    # evaluate joint prob of data in clust k given priors
    H1_pdf = stats.norm()
    probsk = H1_pdf.cdf(np.array([ci.clust, cj.clust]))
    return probsk.prod()

def eval_H2(ci, cj):
    """probability of hypothis 2: data comes from diff distr
    ci and cj are clusters we are checking to merge"""
    return ci.probDatatree * cj.probDatatree
    
    
def prob_clust_k(ci, cj, N, hkparams):
    """calculate the marginal probability of cluster k existing
    This is the 'evidence' in Bayes Rule; called P(D_k | T_k) in paper
    ci and cj are clusters we are checking to merge
    N is total number of data points among all clusters
    hkparams is a dictionary of relevant distribution parameters"""
    ck = Split(ci, cj)
    pik = ck.pi
    probDatatree = (
        ck.pi * eval_H1(ci, cj, **hkparams) +
        (1 - ck.pi) * eval_H2(ci, cj)
        )
