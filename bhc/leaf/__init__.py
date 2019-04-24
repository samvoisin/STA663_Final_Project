import numpy as np


class Leaf:
    """Leaf class contains information for each data point at tier 1 level"""
    
    def __init__(self, pt, clustid, priorParams):
        """initialize Leaf object
        class parameters:
        pt - an individual data point stored as a len 1 np array
        clustid - id number assigned to each leaf and used to identify
        subclusters as algorithm progresses
        priorParams - dictionary of prior parameters (e.g.
        diffuseWishPrior, diffuseNormPrior, clusterConcentrationPrior)
        class attributes:
        tier - level of the tree where the split occurs; leaves are tier 1
        alpha - cluster concentration parameter controlling
        probability of creating a new cluster k; dirichlet scaling factor
        margLik - prob of data under tree (i.e. p(Dk | Tk)); 1 for leaves
        clust - non-nested data points in cluster; leaves have clust size 1
        clustid - id number assigned to each leaf and used to identify
        subclusters as algorithm progresses
        clustsize - number of data points in cluster; leaves have clust size 1
        d - tree density parameter(?)
        pi - probability of cluster k existing
        """
        self.tier = 0
        self.alpha = priorParams["clusterConcentrationPrior"]["alpha"]
        self.priorParams = priorParams
        self.margLik = 1
        self.clust = np.array([pt])
        self.clustid = clustid
        self.clustsize = 1
        self.d = self.alpha
        self.pi = 1
        self.postMergProb = 1.0