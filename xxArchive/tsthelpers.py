import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import random as rnd
import bhc


mu1 = np.zeros(2)
cov1 = np.eye(2)

# number of total draws
draws = 10

rnd.seed(1)
X = rnd.multivariate_normal(mu1, cov1, draws)

x0 = X[0, ]
x1 = X[1, ]
x2 = X[2, ]
x3 = X[3, ]
x4 = X[4, ]
x5 = X[5, ]
x6 = X[6, ]
x7 = X[7, ]
x8 = X[8, ]
x9 = X[9, ]

pmp = []
a = np.linspace(0.001, 0.1, 100)
for i in a:
    allParams = {
    "clusterConcentrationPrior" : {"alpha" : i},
    "diffuseWishPrior" : {"df" : 1, "scale" : 10}, # wishart params
    "diffuseNormPrior" : {"loc" : mu1,
                          "scale" : cov1}, # mvtnormal params
                          }
    l0 = bhc.Leaf(x0, allParams)
    l1 = bhc.Leaf(x1, allParams)
    l2 = bhc.Leaf(x2, allParams)
    l3 = bhc.Leaf(x3, allParams)
    l4 = bhc.Leaf(x4, allParams)
    l5 = bhc.Leaf(x5, allParams)
    l6 = bhc.Leaf(x6, allParams)
    l7 = bhc.Leaf(x7, allParams)
    l8 = bhc.Leaf(x8, allParams)
    l9 = bhc.Leaf(x9, allParams)
    s1 = bhc.Split(l1, l0)
    s2 = bhc.Split(s1, l2)
    s3 = bhc.Split(s2, l3)
    s4 = bhc.Split(s3, l4)
    s5 = bhc.Split(s4, l5)
    s6 = bhc.Split(s5, l6)
    s7 = bhc.Split(s6, l7)
    s8 = bhc.Split(s7, l8)
    s9 = bhc.Split(s8, l9)
    pmp.append(s9.postMergProb)



plt.plot(a, pmp)
plt.ylabel("posterior merge prob")
plt.show()

