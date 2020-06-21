# Ch8 Tree-Based Methods

## Decision Trees

### 1. Regression Trees

The process of building a regression tree:

1. Divide the predictor space -- the set of possible values for $X_1, X_2, ..., X_p$ -- into J distinct and non-overlapping regions, $R_1, R_2, ..., R_J$
2. For every observation that falls into the region $R_j$, we make the same prediction, which is simply the mean of the response values for the training observations in $R_j$.

How to divide space?

$$Minimize_J\ RSS = \sum_{j=1}^J \sum_{i \in R_j} (y_i - \hat{y}_{R_j})^2 $$

But it is computationally infeasible to consider every possible partition, so we take a top-down, greedy approach that is known as recursive binary splitting, which starts from:

$$Minimize_{(j, s)}\ [\sum_{i: x_i \in R_1(j, s)} (y_i - \hat{y}_{R_1})^2 + \sum_{i: x_i \in R_2(j, s)} (y_i - \hat{y}_{R_2})^2]$$

And repeat this process until a stopping criterion is reached like no region contains more than five observations

But it's likely to overfit the data. A smaller tree with fewer splits might lead to lower variance and better interpretation at the cost of a little bias. 

**Tree Pruning**

Cost complexity pruning -- also known as weakest link pruning

1. Use recursive binary splitting to grow a large tree on the training data
2. Apply cost complexity pruning to the large tree in order to obtain a sequence of best subtrees, as a function of $\alpha$
3. Use K-fold cross-validation 

   1. Repeat Steps 1 and 2 on all but the kth fold of the training data.

   2. Evaluate the mean squared prediction error on the data in the left-out kth fold

   3. Average the results for each value of $\alpha$ , and pick $\alpha$ to minimize the average error.
4. Return the subtree from Step 2 that corresponds to the chosen value of $\alpha$

For each value of $\alpha$ there corresponds a subtree $T \subset T_0$ such that minimize

$$\sum_{m=1}^{|T|} \sum_{i: x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$

### 2. Classification Trees

How to calculate RSS?

**Classification Error Rate**

$$E = 1 - max_k (\hat{p}_{mk}) \ in\ mth\ region $$

However, it turns out that classification error is not sufficiently sensitive for tree-growing, and in practice two other measures are preferable.

**Gini Index** (measure of node purity)

$$G = \sum_{k=1}^K \hat{p}_{mk} (1 - \hat{p}_{mk})$$

**Entropy**

$$D = -\sum_{k=1}^K \hat{p}_{mk} \log \hat{p}_{mk}$$

Any of these three approaches might be used when pruning the tree, but the classification error rate is preferable if prediction accuracy of the final pruned tree is the goal.

### 3. Pros and Cons

**Pros:**

* easily explain
* easily handle qualitative predictors without the need to create dummy variables.
* more closely mirror human decision-making

**Cons:**

* less accuracy
* non-robust

## Bagging, Random Forests, Boosting

### 1. Bagging

Bootstrap aggregation,or bagging, is a general-purpose procedure for reducing the variance of a statistical learning method.

Hence a natural way to reduce the variance and hence increase the prediction accuracy of a statistical learning method is to take many training sets from the population, build a separate prediction model using each training set, and average the resulting predictions.

Of course, this is not practical because we generally do not have access to multiple training sets. Instead, we can bootstrap, by taking repeated samples from the (single) training data set. In this approach we generate B different bootstrapped training data sets.

$$\hat{f}_{bag}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$$

B is not a critical parameter with bagging; using a very large value of B will not lead to overfitting. In practice we use a value of B sufficiently large that the error has settled down. 

**Out-of-Bag Error Estimation**

The remaining one-third of the observations not used to fit a given bagged tree are referred to as the out-of-bag (OOB) observations. We can predict the response for the ith observation using each of the trees in which that observation was OOB.

In order to obtain a single prediction for the ith observation, we can average these predicted responses (if regression is the goal) or can take a majority vote (if classification is the goal). This leads to a single OOB prediction for the ith observation. An OOB prediction can be obtained in this way for each of the n observations.

**Variable Importance Measures**

In the case of bagging regression trees, we can record the total amount that the RSS is decreased due to splits over a given predictor, averaged over all B trees.

Similarly, for bagged/RF classification trees, we add up the total amount that the Gini index is decreased by splits over a given predictor, averaged over all B trees.

**Cons:** all of the bagged trees will look quite similar to each other. Hence the predictions from the bagged trees will be highly correlated. Unfortunately, averaging many highly correlated quantities does not lead to as large of a reduction in variance as averaging many uncorrelated quantities. In particular, this means that bagging will not lead to a substantial reduction in variance over a single tree in this setting.

### 2. Random Forests

Random forests provide an improvement over bagged trees by way of a small tweak that decorrelates the trees. 

Each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors. The split is allowed to use only one of those m predictors. Usually set m as $\sqrt{p}$

We can think of this process as decorrelating the trees, thereby making the average of the resulting trees less variable and hence more reliable.

### 3. Boosting

Boosting does not involve bootstrap sampling; instead each tree is fit on a modified version of the original data set, that are grown sequentially. Unlike fitting a single large decision tree to the data, which amounts to fitting the data hard and potentially overfitting, the boosting approach instead learns slowly.

**Boosting Algorithms**

1. Set $\hat{f}(x) = 0$ and $r_i = y_i$ for all i
2. For b = 1, 2, ..., B, repeat:
   1. Fit a tree $\hat{f}^b$ with d splits to (X, r)
   2. update $\hat{f}(x) = \hat{f}(x)  + \lambda\hat{f}^b(x) $
   3. update $r_i = r_i - \lambda\hat{f}^b(x)$
3. Output $\hat{f}(x) = \sum_{b=1}^B\lambda\hat{f}^b(x) $

Boosting has three tuning parameters:

1. The number of trees B. Unlike bagging and random forests, boosting can overfit if B is too large, although this overfitting tends to occur slowly if at all. We use cross-validation to select B.
2. The shrinkage parameter $\lambda$, slows the process down even further, allowing more and different shaped trees to attack the residuals. Typical values are 0.01 or 0.001
3. The number d of splits in each tree, which controls the complexity of the boosted ensemble. More generally d is the interaction depth. Often d = 1 works well, in which case each tree is a stump, consisting of a single split.