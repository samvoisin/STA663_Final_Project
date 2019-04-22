from functools import reduce
from bhc.split import Split
from bhc.helpfncs.hypothesis import posterior_join_k


### HierarchyTree helper functions ###


def get_max_posterior(propck):
    """
    find the unique combination of clusters i and j with highest
    posterior merge probability and return the new cluster
    propck - list of unique proposed combinations of
    clusters i and j
    """
    # find pair with maximum posterior probability of merge
    maxPost = reduce(
        lambda ci, cj: ci if ci.postMergProb > cj.postMergProb else cj,
        propck
    )
    
    return maxPost

###############################################################################


def place_cluster(ck, tree):
        """
        place new cluster k in appropriate tier. If tier does not exist
        this change-of-state method will create it
        """
        if clustk.tier in self.tree.keys():
            self.tree[clustk.tier].update({clustk.clustid : clustk})
        else:
            self.tree[clustk.tier] = self.tree.get(
                clustk.tier,
                {clustk.clustid : clustk}
            )
        
###############################################################################


def update_cluster_list(clist, newck):
    """
    update list of independent clusters as algorithm progresses. this list will
    reduce by one for each iteration as cclusters i and j merge into cluster k
    clist - current list of independent clusters
    newck - newly merged cluster k
    """
    clist.remove(newck.left) # remove cluster i
    clist.remove(newck.right) # remove cluster j
    clist.append(newck) # append new cluster k
    
    return clist


###############################################################################
######################### Prune Tree Helper Functions #########################
###############################################################################


def split_cluster(splobj):
    """
    This is how a cluster is seperated into clusters i and j during pruning
    splobj - Split object defining cluster k
    """
    return splobj.left, splobj.right

###############################################################################


def snip_splits(tier, rk):
    """
    Snip all joins in a tier with posterior merge probabilities < rk
    """
    rejectedJoins = []
    for k, v in tier.items():
        if v.postMergProb < rk:
            rejectedJoins.append(k)
    

