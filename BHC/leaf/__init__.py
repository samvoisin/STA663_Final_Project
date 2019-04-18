import numpy as np

class Leaf:
    """Leaf class contains information for each data point at tier 1 level"""
    
    def __init__(self, pt, alpha):
        """initialize Leaf object
        pt is an individual data point stored as a len 1 np array
        alpha is prior parameter for number of clusters
        tier - level of the tree where the split occurs; leaves are tier 1
		alpha - prior on number of clusters
		tree - nested data points in cluster; leaves have clust size 1
		clust - non-nested data points in cluster; leaves have clust size 1
		clustsize - number of data points in cluster; leaves have clust size 1
		d - tree density parameter(?)
		pi - probability of cluster k existing
        """
        self.tier = 1
        self.alpha = alpha
        self.tree = np.array([pt])
        self.clust = np.array([pt])
        self.clustsize = 1
        self.d = alpha
        self.pi = 1