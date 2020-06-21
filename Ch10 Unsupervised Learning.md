# Ch10 Unsupervised Learning

**The Challenge of Unsupervised Learning：**

* Unsupervised learning tends to be more subjective, and there is no simple goal for the analysis, often performed as part of an exploratory data analysis.

* It can be hard to assess the results obtained from unsupervised learning methods, since there is no universally accepted mechanism for performing cross-validation or validating results on an independent data set.

* But it is often easier to obtain unlabeled data — from a lab instrument or a computer — than labeled data, which can require human intervention.

## Principal Components Analysis

### 1. Basics

PCA produces a low-dimensional representation of a dataset. It finds a sequence of linear combinations of the variables that have maximal variance, and are mutually uncorrelated. 

The first principal component of a set of features is the normalized linear combination of the features

$$Z_1 = \sum_{j=1}^p \phi_{j1} X_j, \  \sum_{j=1}^p \phi_{j1}^2 = 1$$

$\phi_1 = (\phi_{11} \phi_{21} ... \phi_{p1})^T$ is the loadings of the first principal component.

To make variance maximal,

$$maximize_{\phi_{11},..., \phi_{p1}} [\frac{1}{n} \sum_{i=1}^n (\sum_{j=1}^p \phi_{j1} x_{ij})^2], \ \sum_{j=1}^p \phi_{j1}^2 = 1$$

### 2. Another Interpretation

* The first principal component loading vector has a very special property: it defines the line in p-dimensional space that is closest to the n observations (using average squared Euclidean distance as a measure of closeness) 

* The notion of principal components as the dimensions that are closest to the n observations extends beyond just the first principal component. 

* For instance, the first two principal components of a data set span the plane that is closest to the n observations, in terms of average squared Euclidean distance.
* Using this interpretation, together the first M principal component score vectors and the first M principal component loading vectors provide the best M-dimensional approximation (in terms of Euclidean distance) to the ith observation.

### 3. Scaling the Variables

* The results obtained when we perform PCA will also depend on whether the variables have been
  individually scaled.

* Because it is undesirable for the principal components obtained to depend on an arbitrary choice of scaling, we typically scale each variable to have standard deviation one before we perform PCA.

* In certain settings, however, the variables may be measured in the same units. In this case, we might not wish to scale the variables to have standard deviation one before performing PCA.

### 4. PVE

To understand the strength of each component, we are interested in knowing the **proportion of variance explained** (PVE) by each one.

Total variance: $\sum_{j=1}^p Var(X_j) = \sum_{j=1}^p \frac{1}{n} \sum_{i=1}^n x_{ij}^2$

The variance explained by the mth principal component: 

$Var(Z_m) = \frac{1}{n}\sum_{i=1}^n z_{im}^2 = \frac{1}{n} \sum_{i=1}^n (\sum_{j=1}^p \phi_{jm} x_{ij})^2$

**Deciding How Many Principal Components to Use**

We typically decide on the number of principal components required to visualize the data by examining a scree plot, looking for a point at which the proportion of variance explained by each subsequent principal component drops off. This is often referred to as an elbow in the scree plot.

## Clustering Methods

* PCA looks to find a low-dimensional representation of the observations that explain a good fraction of the variance;

* Clustering looks to find homogeneous subgroups among the observations.

### 1. K-Means Clustering

The idea behind K-means clustering is that a good clustering is one for which the within-cluster variation is as small as possible.

$$minimize_{C_1, ..., C_K} \sum_{k=1}^K W(C_k)$$

There are many possible ways to define this concept, but by far the most common choice involves squared Euclidean distance. $W(C_k) = \frac{1}{|C_k|} \sum_{i, i' \in C_k} \sum_{j=1}^p (x_{ij} - x_{i'j})^2 = 2\sum_{i \in C_k} \sum_{j=1}^p (x_{ij} - \bar{x}_{kj})^2$

A local optimum algorithm

1. Randomly assign a number, from 1 to K, to each of the observations. These serve as initial cluster assignments for the observations.
2. Iterate until the cluster assignments stop changing:
   * For each of the K clusters, compute the cluster centroid. The kth cluster centroid is the vector of the p feature means for the observations in the kth cluster.
   * Assign each observation to the cluster whose centroid is closest (where closest is defined using Euclidean distance).

Because the K-means algorithm finds a local rather than a global optimum, the results obtained will depend on the initial (random) cluster assignment of each observation in Step 1. For this reason, it is important to run the algorithm multiple times from different random initial configurations.

**Cons:** need to pre-specify K

### 2. Hierarchical

We describe bottom-up or agglomerative clustering. This is the most common type of hierarchical clustering, and refers to the fact that a **dendrogram** is built starting from the leaves and combining clusters up to the trunk.

1. Begin with n observations and a measure (such as Euclidean distance) of all the $C_n^2$ pairwise dissimilarities. 
2. For i = n, n-1, ..., 2
   * Examine all pairwise inter-cluster dissimilarities among the i clusters and identify the pair of clusters that are least dissimilar (that is, most similar). Fuse these two clusters. The dissimilarity between these two clusters indicates the height in the **dendrogram** at which the fusion should be placed.
   * Compute the new pairwise inter-cluster dissimilarities among the i − 1 remaining clusters.

| Linkage  | Description                                                  |
| -------- | ------------------------------------------------------------ |
| Complete | Maximal inter-cluster dissimilarity.                         |
| Single   | Minimal inter-cluster dissimilarity.                         |
| Average  | Mean inter-cluster dissimilarity.                            |
| Centroid | Dissimilarity between the centroid of 2 clusters. But can result in undesirable inversions. |

 Average and complete linkage are generally preferred over single linkage, as they tend to yield more balanced dendrograms. Centroid linkage is often used in genomics, but suffers from a major drawback in that an inversion can occur, whereby two clusters are fused at a height below either of the individual clusters in the dendrogram. 

### 3. Choice of Dissimilarity Measure

Correlation-based distance considers two observations to be similar if their features are highly correlated, even though the observed values may be far apart in terms of Euclidean distance. Correlation-based distance focuses on the shapes of observation profiles rather than their magnitudes.