from functools import reduce
from bhc.split import Split
from bhc.helpfncs.hypothesis import posterior_join_k


### HierarchyTree helper functions ###


###############################################################################
######################### grow_tree Helper Functions ##########################
###############################################################################


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
    tree[ck.tier] = {ck.clustid : ck}
        
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
######################### prune_tree Helper Functions #########################
###############################################################################


def find_bad_merges(tier, rk):
    """
    find merges where posterior merge probability < rk
    tier - integer identifying a tier in tree
    rk - posterior merge probability cut threshold; defaults to 0.5
    """
    badMerges = [] # initialize list for unjustified merge ids
    for cid, s in tier.items():
        if s.postMergProb < rk:
            badMerges.append(cid) # append clustid to bad merge list
    
    return badMerges
    
###############################################################################


def snip_splits(tier, cutpts):
    """
    cut all points in a tier identified as unjustified merge
    tier - integer identifying a tier in tree
    cutpts - list of points where cluster joins are to be seperated
    """
    for cut in cutpts:
        tier.update({tier[cut].left.clustid : tier[cut].left})
        tier.update({tier[cut].right.clustid : tier[cut].right})
        tier.pop(cut)
            
###############################################################################


def clear_trimmings(tier, rk):
    """
    remove snipped clusters from a tier. This function should only be run
    after the entire tree has been pruned. 
    """
    trimmings = []
    for k, splt in tier.items():
        if splt.postMergProb < rk:
            trimmings.append(k)
    
    for t in trimmings:
        tier.pop(t)
