from pandas import DataFrame
from numpy import float32

from itertools import combinations
from . import split
from . import leaf
from . import hierarchytree_fncs


class HierarchyTree:
    """
    Primary Bayesian Hierarchical Clustering object; HierarchyTree contains
    all of the Leaf and Split objects. The tree is complete when all points
    have been agglomerated into a single hierarchical cluster.

    The grow_tree method provides the mechanism for the clustering algorithm
    The prune_tree method 'snips' merges that have a posterior merge
    probability below a given threshhold (default = 0.5).
    The generate_clust_dataframe method will create a new attribute within
    the HierarchyTree instance with the dimensions of the original data
    and a new column indicating the highest level cluster.
    """

    def __init__(self, X, allParams, family = "norm-invwish"):
        """
        Initialize HierarchyTree object
        class parameters:
        X - numpy array of vectors or points to be clustered
        allParams - dictionary of prior parameters (e.g.
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
        self.data = X
        self.clustCount = X.shape[0]
        self.leaves = {
            n : Leaf(i, n, allParams, self.family) for n, i in enumerate(X)
            }
        self.currTier = self.leaves
        self.tree = {0 : self.leaves} # tier 0 is Leaf tier
        self.clusterList = [
            Leaf(i, n, allParams, self.family) for n, i in enumerate(X)
            ]
        self.tierList = [t for t in self.tree.keys()]
        self.ntiers = len(self.tierList)


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

        # generate list of tier numbers in tree
        self.tierList = [t for t in self.tree.keys()]
        self.ntiers = len(self.tierList)


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

        # update list of tier numbers in tree
        self.tierList = [t for t in self.tree.keys()]

        # remove tiers containing only one leaf
        snglLeafTiers = []
        for tier, branch in self.tree.items():
            if len(branch) == 1:
                for cid, spl in branch.items():
                    if spl.clustsize == 1:
                        snglLeafTiers.append(tier)

        for st in snglLeafTiers:
            self.tree.pop(st)

        # update list of tier numbers in tree
        self.tierList = [t for t in self.tree.keys()]
        self.ntiers = len(self.tierList)


    def tier_summary(self, tiernum = "top"):
        """
        summarize a tier
        """
        print(f"Summary for tier {tiernum}:")
        print("-------------------------------")
        print(f"Number of clusters: {len(self.tree[tiernum].values())}")
        for n, c in enumerate(self.tree[tiernum].values()):
            print(f"  Cluster {n + 1} size: {c.clustsize}")
            print(f"\t Posterior merge probability: {c.postMergProb:.2}")
        print("\n")


    def tree_summary(self, ntiers = 0):
        """
        summarize tree structure from highest tier to lowest
        ntiers is number of tiers to show - default value
         will return the entire tree
        """
        tiers = [t for t in self.tree.keys()]
        tiers.reverse() # ordered, descending integers for tiers

        if ntiers == 0:
            ntiers = self.ntiers

        ctr = 0
        for n in tiers:
            self.tier_summary(n)
            ctr += 1
            if ctr == ntiers:
                break


    def generate_clust_frame(self):
        """
        generate pandas DataFrame object with a column of indicators for
        which cluster the vector in that row belongs to
        """
        # generate column labels
        labels = ["Dim_" + str(i) for i in range(self.data.shape[1])]
        self.clustDF = DataFrame(self.data, columns = labels)
        # initialize empty set to track vectors whose id has been accounted for
        idacctfor = set()
        c = 0 # cluster indicator

        # initialize clustnum column with each vectors leaf level id
        for n, v in self.leaves.items():
            self.clustDF.loc[n, "clustnum"] = v.idset

        # iterate through each tier
        for t in self.tierList:
            # iterate through each Split in tier t
            for k, v in self.tree[t].items():
                if t == 0:
                    pass #print(v.idset)
                # the intersection of the previously clustered ids and the
                # current cluster v is empty when points have not been labeled
                if len(idacctfor.intersection(v.idset)) == 0:
                    idacctfor = idacctfor.union(v.idset)
                else:
                    # label points 
                    for i in idacctfor.intersection(v.idset):
                        # assign cluster number for each point in cluster
                        # this goes from bottom to top of tree
                        self.clustDF.loc[i, "clustnum"] = c
                    c += 1 # update to provide a new value for the next cluster

