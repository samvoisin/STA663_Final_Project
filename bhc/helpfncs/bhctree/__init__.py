from functools import reduce
from itertools import combinations
from bhc.split import Split
from bhc.helpfncs.hypothesis import posterior_join_k


def get_max_posterior(tier):
    """
    find the unique combination of clusters with highest
    posterior merge probability and return combination indices
    tier - current top tier of tree; a dictionary where keys are
    unique cluster combinations
    """
    # create set of all possible combinations of sub-clusters
    # note: it.combinations object deleted w/ no warning after iters
    clusts = {
        c : posterior_join_k(tier[c[0]], tier[c[1]]) \
        for c in combinations(tier.keys(), 2)
    }
    
    # find pair with maximum posterior probability of merge
    maxPost = reduce(
        lambda ci, cj: ci if clusts[ci] > clusts[cj] else cj,
        clusts.keys()
    )
    
    return maxPost

###############################################################################


def form_new_tier(tier, ck):
    """
    form a new tier combining two sub-clusters (Leaf or Split)
    into a new Split object
    tier - current top HierarchyTree tier
    ck - tuple of indices indicating sub-clusters to merge
    """
    # form new cluster k
    clusterk = Split(tier[ck[0]], tier[ck[1]])
    
    newTier = {k : v for k, v in tier.items() if k not in ck}
    # insert cluster k into newTier w/ smallest num identifier
    newTier[min(ck)] = clusterk

    return newTier