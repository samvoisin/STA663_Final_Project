import numpy as np
from scipy.special import gamma
from bhc.helpfncs.hypothesis import marginal_clust_k

class Split:
    """
    Split class stores all information about a given split
    clusti and clustj are two clusters lower in the hierarchy
    which are joined to form new cluster k
    """
    
    def __init__(self, clusti, clustj):
        """
        initialize Split object
        class parameters:
        left - a Split or Leaf object from left tree below
        right - a Split or Leaf object from right tree below
        class attributes:
        priorParams - prior parameters inherited from previous Split/ Leaf
        alpha - cluster concentration parameter controls
        prob of creating the new cluster k; inverited from previous Split/ Leaf
        tier - level of the tree where the split occurs
		alpha - prior on number of clusters; unchanged for entire tree (prior)
		clust - non-nested data points in cluster
		clustsize - number of data points in cluster
		d - tree depth parameter(?)
		pi - prob of merging clusters i and j
        probDataTree - prob of data under tree (i.e. p(Dk | Tk))
        """
        self.left = clusti
        self.right = clustj
        self.priorParams = clusti.priorParams
        self.alpha = clusti.alpha
        self.tier = max(clusti.tier, clustj.tier) + 1
        self.clustsize = clusti.clustsize + clustj.clustsize
        self.clust = np.vstack([clusti.clust, clustj.clust])
        
        # calculate new d_k
        self.d = (
        	self.alpha * gamma(self.clustsize) +
        	clusti.d * clustj.d
        	)
        
        # calculate new pi_k
        self.pi = self.alpha * gamma(self.clustsize) / self.d

        # calculate marginal likelihood for this cluster (i.e. p(Dk | Tk))
        self.margLik = marginal_clust_k(self.left, self.right)
















