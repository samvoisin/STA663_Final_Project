# Bayesian Hierarchical Clustering

Bayesian Hierarchical Clustering (BHC) is an agglomerative tree-based method for identifying underlying population structures ("clusters"). BHC was introduced by K. Heller and Z. Ghahramani as a way to approximate the more computationally intensive infinite Gaussian mixture model.

The advantage of these models over their counterparts lies in the fact that an *ex ante* number of clusters does not need to be specified. Instead, the Bayesian paradigm allows for regularized flexibility via a prior placed on the cluster concentration parameter $\alpha$. This module is build using bject-oriented programing (OOP) methodologies found in python to build a module that can be used similar manner to the popular scikit-learn library. As with many Bayesian methods, the increased flexibility of BHC comes at a computational cost and the increased risk of poor results due to misspecified priors. 

---
 
The core of the BHC algorithm relies on a Bayesian hypothesis test in which two alternatives are compared:

1) $H_1$ is the hypothesis that two clusters $D_i$ and $D_j$ were generated from the same distribution $p(x | \theta)$ with the prior distribution for $\theta$ being $p(\theta | \beta)$. The probability of clusters $i$ and $j$ being generated from the same distribution is defined as $p(D_k|H_1)$. The posterior for this hypothesis is:

$$
r_k = \frac{\pi_k p(D_k|H_1)}{\pi_k p(D_k|H_1) + (1 - \pi_k) p(D_i|T_i)p(D_j|T_j)}
$$

Note that $\pi_k$ is the prior probability of a merge occuring for clusters $i$ and $j$. This makes the denominator of this expression the Bayesian *evidence*.

2) $H_2$ is the hypothesis that the two clusters $D_i$ and $D_j$ were generated from two independent distributions and therefore should *not* be joined together as cluster $D_k$. The probability of $H_2$ is calculated as $p(D_k|H_2) = p(D_i|T_i)p(D_j|T_j)$ where $T_i$ and $T_j$ are the subclusters being examined.

All existing clusters are compared and joined based on the cluster with the highest posterior merge probability $r_k$. In our module this iterative action of comparing and merging clusters occurs in the `HierarchyTree` instance.

### The `HierarchyTree` Class

The HierarchyTree class is the primary object in the bhc module. `HierarchyTree` contains all of the `Leaf` and `Split` class instances. The tree is complete when all points have been agglomerated into a single hierarchical cluster. The `grow_tree` method provides the mechanism for the clustering algorithm and the `prune_tree` method removes unjustified clusters and takes the appropriate steps to remove unecessary tiers from the tree.

`HierarchyTree` is initialized by passing in the data matrix `X`, the parameters of the prior distributions (see below), and the family of distributions. Currently, support for the Multivariate Normal-Inverse Wishart family exists with support for the Beta-Binomial and Multinomial-Dirichlet planned as the next steps in development of this module. The `HierarchyTree` is "grown" using the `grow_tree` method. This method is analogous to the `fit` method for a class in `sklearn`.

The `generate_clust_frame` method will create an attribute in the `HierarchyTree` instance called `clustDF` which contains a pandas `DataFrame` object with the each row aligning to an input data point/ vector from the original data matrix `X` and an additional column designating the highest level cluster for a given data point/ vectors. **Note:** if this attribute is run before the `prune_tree` method, all data points will be in the same cluster. This is the nature of the BHC algorithm.

##### Prior Parameters

The bhc module currently supports the multivariate Gaussian-inverse Wishart and Beta-Bernoulli conjugate families of distributions. Support for the Dirichlet-Multinomial conjugate family is in development.

Prior parameters for the multivariate Gaussian-inverse Wishart case should be structured as follows:

    allParams = {
        "clusterConcentrationPrior" : {"alpha" : 2},
        "diffuseInvWishPrior" : {"df" : 10,
                                 "scale" : np.eye(2)},
        "diffuseNormPrior" : {"loc" : np.zeros(2),
                              "scale" : np.eye(2),
                              "meanscale" : 1}
    }





### Background

State the research paper you are using. Describe the concept of the algorithm and why it is interesting and/or useful. If appropriate, describe the mathematical basis of the algorithm. Some potential topics for the backgorund include:

- What problem does it address? 
- What are known and possible applications of the algorithm? 
- What are its advantages and disadvantages relative to other algorithms?
- How will you use it in your research?

### Description of algorithm

First, explain in plain English what the algorithm does. Then describes the details of the algorihtm, using mathematical equations or pseudocode as appropriate. 

### Describe optimization for performance

First implement the algorithm using plain Python in a straightforward way from the description of the algorihtm. Then profile and optimize it using one or more apporpiate mathods, such as:

1. Use of better algorithms or data structures
2. Use of vectorization
3. JIT or AOT compilation of critical functions
4. Re-writing critical functions in C++ and using pybind11 to wrap them
5. Making use of parallelism or concurrency
6. Making use of distributed compuitng

Document the improvemnt in performance with the optimizations performed.

### Applications to simulated data sets

Are there specific inputs that give known outuputs (e.g. there might be closed form solutions for special input cases)? How does the algorithm perform on these? 

If no such input cases are available (or in addition to such input cases), how does the algorithm perform on simulated data sets for which you know the "truth"? 

### Applications to real data sets

Test the algorithm on the real-world examples in the orignal paper if possible. Try to find at least one other real-world data set not in the original paper and test it on that. Describe and interpret the results.

### Comparative analysis with competing algorithms

Find two other algorihtms that addresss a similar problem. Perform a comparison - for example, of accurary or speed. You can use native libraires of the other algorithms - you do not need to code them yourself. Comment on your observations. 
 - Dirichlet Process Model
 - Hierarchical Clustering (frequentist)


### Discussion/conclusion

Your thoughts on the algorithm. Does it fulfill a particular need? How could it be generalized to other problem domains? What are its limiations and how could it be improved further?

### References/bibliography

Make sure you cite your sources.

## Code

The code should be in a public GitHub repository with:

1. A README file
2. An open source license
3. Source code
4. Test code
5. Examples
6. A reproducible report

The package should be downloadable and installable with `python setup.py install`, or even posted to PyPI adn installable with `pip install package`.


## Rubric

Each item is worth 10 points, but some sections will give up to 10 bonus points if done really well. Note that the "difficulty factor" of the chosen algorithm will be factored into the grading. 

1. Is the abstract, background and discussion readable and clear? (10-20 points)
2. Is the algorithm description clear and accurate? (10-20 points)
3. Has the algorihtm been optimized? (10-20 points)
4. Are the applicaitons to simulated/real data clear and useful? (10-20 points)
5. Was the comparative analysis done well? (10-20 points points)
6. Is there a well-maitnatined Github repository for the code? (10 points)
7. Is the document show evidenc of literate programming? (10 points)
8. Is the analyiss reproducible? (10 points)
9. Is the code tested? Are examples provided? (10 points)
10. Is the package easily installable? (10 points)

