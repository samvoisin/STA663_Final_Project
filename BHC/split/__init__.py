import numpy as np
from scipy.special import gamma
from helpfcns import prob_clust_k

class Split:
    """Split class stores all information about a given split"""
    
    def __init__(self, clusti, clustj):
        """initialize Split object
        left - a Split or Leaf object from left tree below
        right - a Split or Leaf object from right tree below
        alpha - prior parameter for number of clusters
        tier - level of the tree where the split occurs
		alpha - prior on number of clusters; unchanged for entire tree (prior)
		tree - nested data points in cluster
		clust - non-nested data points in cluster
		clustsize - number of data points in cluster
		d - tree density parameter(?)
		pi - prob of merging clusters i and j
        probDataTree - prob of data under tree (i.e. p(Dk | Tk))"""
        self.left = clusti
        self.right = clustj
        self.alpha = clusti.alpha
        self.tier = clusti.tier + clustj.tier
        self.clustsize = clusti.clustsize + clustj.clustsize
        self.tree = np.array([clusti.tree, clustj.tree])
        self.clust = np.hstack([clusti.clust, clustj.clust])
        
        # calculate new small d_k
        self.d = (
        	self.alpha * gamma(self.clustsize) +
        	clusti.d * clustj.d
        	)
        
        # calculate new pi_k
        self.pi = self.alpha * gamma(self.clustsize) / self.d

        #calculate new prob of data under tree (i.e. p(Dk | Tk))
        self.probDataTree = prob_clust_k(clusti, clustj, self.clustsize) # needs to be a parameter **kwarg here
