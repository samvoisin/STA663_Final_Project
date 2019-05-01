import numpy as np
import scipy.linalg as la
from scipy import stats
from scipy.special import gamma, loggamma

def beta_bern_H1(ci, cj):
    """
    Evaluate hypothesis 1 for merging two clusters ci and cj.
    This function assumes Bernoulli-Beta conjugacy.
    Will need to generalize later
    ci and cj - clusters being tested for merge
    
    Beta prior hyperparams:
    alpha - alpha-1 successes
    beta - beta-1 failures
    
    
    Bernoulli likelihood
    
    hkparams - dictionary of prior parameters (e.g.
    alphaPrior, betaPrior)
    """

    # abbreviated keyword arguments for brevity in calcs;
    # not the most efficient thing to do, but it's here for now...
    X = np.vstack([ci.clust, cj.clust])
    N, k = X.shape
    alpha = ci.priorParams["alphaPrior"]["succ"]
    beta = ci.priorParams["betaPrior"]["fail"]
    
    ## version of eval_H1 from Heller appendix
    # posterior precision matrix
    m = np.sum(X, axis = 0) # needs to be done by component k
    
    # components of p(D_k | H_1)
    numer = [(loggamma(alpha + beta) + loggamma(alpha + m) + loggamma(beta + N - m)) for d in range(1, k + 1)]
    numer = np.sum(numer)
    denom = [(loggamma(alpha) + loggamma(beta) + loggamma(alpha + beta + N)) for d in range(1, k + 1)]
    denom = np.sum(denom)
    
    MarginalLikelihood = numer - denom
    
    return MarginalLikelihood