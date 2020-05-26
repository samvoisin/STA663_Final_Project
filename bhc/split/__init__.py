import numpy as np
from scipy.special import gamma
from bhc.helpfncs.hypothesis import marginal_clust_k, posterior_join_k


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
        alpha - cluster concentration parameter controlling
        probability of creating a new cluster k; dirichlet scaling factor
        clust - non-nested data points in cluster
        clustid - tuple of id numbers used to identify subclusters as
        algorithm progresses; see "tree consistent" in BHC paper
        clustsize - number of data points in cluster
        d - tree depth parameter
        pi - prob of merging clusters i and j
        margLik - prob of data under tree (i.e. p(Dk | Tk))
        postMergProb - posterior probability for cluster k; referred to as rk
        in Heller and Ghahramani BHC paper
        """
        self.left = clusti
        self.right = clustj
        self.family = clusti.family
        self.clustid = (self.left.clustid, self.right.clustid)
        self.idset = clusti.idset.union(clustj.idset)
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
        #if np.isnan(self.alpha * gamma(self.clustsize) / self.d) == True:
        #    self.pi = 1.0
        #else:
        self.pi = self.alpha * gamma(self.clustsize) / self.d


        # calculate marginal likelihood for this cluster (i.e. p(Dk | Tk))
        self.pH1, self.margLik = marginal_clust_k(self.left, self.right)

        # calculate posterior merge probability for this cluster (i.e. rk)
        self.postMergProb = posterior_join_k(self.left, self.right)
