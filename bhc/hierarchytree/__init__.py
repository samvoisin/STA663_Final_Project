from itertools import combinations
from bhc.split import Split
from bhc.leaf import Leaf
from bhc.helpfncs.hierarchytree import *


class HierarchyTree:
    """
    Primary Bayesian Hierarchical Clustering object; HierarchyTree contains
    all of the Leaf and Split objects. The tree is complete when all points
    have been agglomerated into a single hierarchical cluster.

    The grow_tree method provides the mechanism for the clustering algorithm.
    """

    def __init__(self, X, allParams, family = "norm-invwish"):
        """
        Initialize HierarchyTree object
        class parameters:
        X - numpy array or pandas DataFrame to be clustered
        priorParams - dictionary of prior parameters (e.g.
        diffuseWishPrior, diffuseNormPrior, clusterConcentrationPrior)
		family - conjugate family to be used in posterior
		calculations. Currently supported families are:
			- norm-invwish : mutlivariate-normal inverse-wishart
			- beta-bern : beta-bernoulli
        class attributes:
        clustCount - number of non-joined trees; incremented -1 each iter
        leaves - initial clusters of single points stored in dictionary
        currTier - current tier of tree being tested
        clusterList - list of unjoined/ independent clusters remaining to
        be tested for merge
        tree - tree grown by the bhc algorithm; dictionary of tiers;
        key is tier number and value is tier
        """
        self.family = family
        self.clustCount = X.shape[0]
        self.leaves = {
            n : Leaf(i, n, allParams, self.family) for n, i in enumerate(X)
            }
        self.currTier = self.leaves
        self.tree = {0 : self.leaves} # tier 0 is Leaf tier
        self.clusterList = [
            Leaf(i, n, allParams, self.family) for n, i in enumerate(X)
            ]
        self.tierList = [self.tree.keys()]


    def grow_tree(self):
        """
        Grow the tree over the maximum possible number of iterations.
        This means all points will be joined into a single cluster at
        the top level of the hierarchy. See `prune_tree` method.
        """
        while len(self.clusterList) > 1:
            # proposed clusters
            propClusts = [
                Split(c[0], c[1]) for c in combinations(self.clusterList, 2)
            ]
            clustk = get_max_posterior(propClusts) # highest posterior cluster
            if clustk.tier in self.tree.keys():
                self.tree[clustk.tier].update({clustk.clustid : clustk})
            else:
                self.tree[clustk.tier] = self.tree.get(
                    clustk.tier, {clustk.clustid : clustk}
                )
            self.clusterList = update_cluster_list(self.clusterList, clustk)


    def prune_tree(self, rk = 0.5):
        """
        Cut the tree at points where the posterior merge probability < rk
        starting from top tier and going to bottom
        rk - posterior merge probability cut threshold; defaults to 0.5
        """
        tiers = [t for t in self.tree.keys()]
        tiers.reverse() # ordered, descending integers for tiers
        for t in tiers:
            cutPoints = find_bad_merges(self.tree[t], rk) # see HT helpers
            snip_splits(self.tree[t], cutPoints) # see HT helpers

        # remove snipped Split objects
        emptyTiers = []
        for t in tiers:
            clear_trimmings(self.tree[t], rk) # see HT helpers
            # track empty tiers
            if len(self.tree[t].keys()) == 0:
                emptyTiers.append(t)

        # drop tiers containing no Split objects
        for et in emptyTiers:
            self.tree.pop(et)



    def tier_summary(self, tiernum):
        """
        summarize a tier
        """
        print(f"Summary for tier {tiernum}:")
        print("-------------------------------")
        print(f"Number of clusters: {len(self.tree[tiernum].values())}")
        for n, c in enumerate(self.tree[tiernum].values()):
            print(f"  Cluster {n} size: {c.clustsize}")
            print(f"\t Posterior merge probability: {c.postMergProb:.2}")
        print("\n")

    def tree_summary(self):
        """
        summarize tree structure from highest tier to lowest
        """
        tiers = [t for t in self.tree.keys()]
        tiers.reverse() # ordered, descending integers for tiers
        for n in tiers:
            self.tier_summary(n)
