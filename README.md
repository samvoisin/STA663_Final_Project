# Final Project: Bayesian Hierarchical Modeling - STA 663 - Statistical Computing and Computation - Spring 2019

Deadline 30th April 2018 at 11:59 PM

**Remaining To-Dos:**
! Comparison against other models (speed and accuracy)
! 
1) Model Summary - ideally something that works for both pre and post pruned trees
2) Plots if possible
3) Support for Beta-Binom and Dirichlet-Multinom
4) Prediction
5) More test cases

## Paper

The paper should have the following:

### Title

Should be consise and informative.

### Abstract

250 words or less. Identify 4-6 key phrases.

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

