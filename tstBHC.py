###############################################################################
###############################################################################
############### Test Script for Bayesian Hierarchical Clustering ##############
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

empX = clustData.iloc[1:10,0:2].values
empXtX = empX.T @ empX

allParams = {
    "clusterConcentrationPrior" : {"alpha" : 10},
    "diffuseWishPrior" : {"df" : 1, "scale" : 1}, # wishart params
    "diffuseNormPrior" : {"loc" : np.mean(clustData.iloc[:,0:2], axis = 0),
                          "scale" : empXtX}, # mvtnormal params
}

### subset of test data points ###

x = empX[0, :]
y = empX[1, :]
z = empX[2, :]

# assign some leaves
lx = bhc.Leaf(x, allParams)
ly = bhc.Leaf(y, allParams)
lz = bhc.Leaf(z, allParams)

# testing merges x and y
print("\nTest merge x and y")
xyH1 = bhc.eval_H1(lx, ly)
print(f"Hypothesis 1: {xyH1}")
xyH2 = bhc.eval_H2(lx, ly)
print(f"Hypothesis 2: {xyH2}")
xyMarg = bhc.posterior_join_k(lx, ly)
print(f"Posterior probability for cluster k: {xyMarg}\n")

# testing merges x and z
print("\nTest merge x and z")
xzH1 = bhc.eval_H1(lx, lz)
print(f"Hypothesis 1: {xzH1}")
xzH2 = bhc.eval_H2(lx, lz)
print(f"Hypothesis 2: {xzH2}")
xzMarg = bhc.posterior_join_k(lx, lz)
print(f"Posterior probability for cluster k: {xzMarg}\n")

# testing merges z and y
print("\nTest merge z and y")
zyH1 = bhc.eval_H1(lz, ly)
print(f"Hypothesis 1: {zyH1}")
zyH2 = bhc.eval_H2(lz, ly)
print(f"Hypothesis 2: {zyH2}")
zyMarg = bhc.posterior_join_k(lz, ly)
print(f"Posterior probability for cluster k: {zyMarg}\n")

### x and z have highest posterior prob ###

xzSplit = bhc.Split(lx, lz)

print("Joining leaf x and leaf z...\n")
print("xzSplit attributes:")
print(f"left: {xzSplit.left}")
print(f"right: {xzSplit.right}")
print(f"alpha: {xzSplit.alpha}")
print(f"tier: {xzSplit.tier}")
print(f"clustsize: {xzSplit.clustsize}")
print(f"tree:\n {xzSplit.tree}")
print(f"clust:\n {xzSplit.clust}\n\n")
print(f"tree dimensions: {xzSplit.tree.shape}")
print(f"cluster dimensions: {xzSplit.clust.shape}")

print(f"Marginal likelihood for xzSplit: {xzSplit.margLik}")

# testing merges xz and y
print("\nTest merge xz and y")
xzyH1 = bhc.eval_H1(xzSplit, ly)
print(f"Hypothesis 1: {xzyH1}")
xzyH2 = bhc.eval_H2(xzSplit, ly)
print(f"Hypothesis 2: {xzyH2}")
xzyPost = bhc.posterior_join_k(xzSplit, ly)
print(f"Posterior probability for cluster k: {xzyPost}\n")