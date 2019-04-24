###############################################################################
###############################################################################
############### Test Script for Bayesian Hierarchical Clustering ##############
######################### testing HierarchyTree class #########################
###############################################################################
###############################################################################

import numpy as np
import scipy as sc
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import linalg as la
from scipy import random as rnd
import matplotlib.pyplot as plt

import bhc


### generate some random data ###

# number of total draws
draws = 60

# bivariate gaussian params
mu1 = np.zeros(2)
cov1 = np.eye(2)

mu2 = np.array([5, 3])
cov2 = np.eye(2) * 2

mu3 = np.array([8, 12])
cov3 = np.array([3.4, 0, 0, 5.1]).reshape(2, 2)

# multinom params
p1 = 0.2
p2 = 0
p3 = 1 - p2 - p1

# random draws
rnd.seed(1)
knum = rnd.multinomial(draws, (p1, p2, p3))
gaus1 = rnd.multivariate_normal(mu1, cov1, knum[0])
gaus2 = rnd.multivariate_normal(mu2, cov2, knum[1])
gaus3 = rnd.multivariate_normal(mu3, cov3, knum[2])

# join columns into dataframe
x1 = pd.Series(np.r_[gaus1[:, 0], gaus2[:, 0], gaus3[:, 0]])
x2 = pd.Series(np.r_[gaus1[:, 1], gaus2[:, 1], gaus3[:, 1]])
c = pd.Series(np.r_[np.zeros(knum[0]), np.ones(knum[1]), np.ones(knum[2]) * 2])
dat = {"x1" : x1, "x2" : x2, "c" : c}
clustData = pd.DataFrame(dat)

# plot clusters
#plt.scatter(clustData["x1"], clustData["x2"], c = clustData["c"])
#plt.show()


### Set Clustering Prior Paramteres ###

# Gaussian Distribution - loc is mean; scale is sd
# Gamma Distribution - a is shape; scale is rate; leave loc at 0

empX = clustData.iloc[: , 0:2].values
empXtX = empX.T @ empX

#plt.scatter(empX[: , 0], empX [: , 1])
#plt.show()


allParams = {
    "clusterConcentrationPrior" : {"alpha" : 2},
    "diffuseInvWishPrior" : {"df" : 10, "scale" : empXtX}, # inv wishart params
    "diffuseNormPrior" : {"loc" : np.mean(clustData.iloc[:,0:2], axis = 0),
                          "scale" : empXtX,
                          "meanscale" : 1}, # mvtnormal params
}


### subset of test data points ###

ht = bhc.HierarchyTree(empX, allParams)


ht.grow_tree()

print(f"\nGrowing tree...")

ht.prune_tree()

print("\nPruning tree...")

print("\nTesting reporting...")
ht.tier_summary(2)

ht.tree_summary()




