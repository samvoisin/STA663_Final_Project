from bhc.split import Split
from bhc.leaf import Leaf
from bhc.helpfncs.bhctree import form_new_tier, get_max_posterior

### I think we may need some sort of nested structure here. not sure if tier-ing is enough...
class HierarchyTree:
    """
    Primary Bayesian Hierarchical Clustering object; HierarchyTree contains
    all of the Leaf and Split objects. The tree is complete when all points
    have been agglomerated into a single hierarchical cluster.

    The grow_tree method provides the mechanism for the clustering algorithm.
    """
    
    def __init__(self, X, allParams):
        """
        Initialize HierarchyTree object
        class parameters:
        X - numpy array or pandas DataFrame to be clustered
        priorParams - dictionary of prior parameters (e.g.
        diffuseWishPrior, diffuseNormPrior, clusterConcentrationPrior)
        class attributes:
        clustCount - number of non-joined trees; incremented -1 each iter
        leaves - initial clusters of single points stored in dictionary
        tierNum - current level of tree (int); incremented +1 each iter
        currTier - current tier of tree being tested
        tree - tree grown by the bhc algorithm; dictionary of tiers;
        key is tier number and value is tier
        """
        self.clustCount = X.shape[0]
        self.leaves = {n : Leaf(i, allParams) for n, i in enumerate(X)}
        self.tierNum = 0
        self.currTier = self.leaves
        self.tree = {self.tierNum : self.leaves}
    
    def grow_tree(self):
        """
        Grow the tree over the maximum possible number of iterations.
        This means all points will be joined into a single cluster at
        the top level of the hierarchy. See `prune_tree` method.
        """
        while self.clustCount > 1:
            ckid = get_max_posterior(self.currTier) # clusters i and j id nums
            self.currTier = form_new_tier(self.currTier, ckid)
            self.tierNum += 1 # iterate up the tree
            self.tree[self.tierNum] = self.currTier # add new tier to tree
            self.clustCount -= 1 # remove 1 from number of clusters
        
    def prune_tree(self, rk = 0.5):
        """
        Cut the tree at points where the posterior merge probability < rk
        rk - posterior merge probability cut threshold; defaults to 0.5
        """
        pass
            
