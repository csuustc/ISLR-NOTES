# Ch6 Linear Model Selection and Regularization

**Why we need alternative fitting procedures instead of least squares?**

- *Prediction Accuracy*: If relationship is approximately linear, the least squares estimates will have low bias. If n >> p, then it tend to also have low variance. *However*, if n is not much larger then p, there will be higher variance and overfitting. And if p > n, then there is no unique least squares estimate and the variance is infinite. By constraining or shrinking the estimated coefficients, we can often substantially reduce the variance at the cost of a negligible increase in bias.
- *Model Interpretability*: By removing irrelevant variables, we can get a model is more easily interpreted.

## Subset Selection

### 1. Best Subset Selection

Select the best model from all the $2^p$ models:

1. Start from $M_0$, which includes 0 predictor
2. For k = 1, 2, ..., p: Fit all $C_p^k$ models contain exactly k predictors, then pick the best model with smallest RSS, and call it $M_k$ 
3. Select best model from $M_0, M_1, ..., M_p$ using cross-validated prediction error, AIC, BIC, or adjusted $R^2$

**Limitation:** computationally infeasible for p > 40; also, The larger the search space, the higher the chance of finding models that look good on the training data, even though they might not have any predictive power on future data. Thus an enormous search space can lead to overfitting and high variance of the coefficient estimates.

### 2. Stepwise Selection

**Forward Stepwise Selection**

1. Start from $M_0$ 
2. For k = 0, 1, 2, ..., p-1: Fit all p - k models upon $M_k$ with one additional predictor and choose the best as $M_{k+1}$ 
3. Select best model from $M_0, M_1, ..., M_p$ using cross-validated prediction error, AIC, BIC, or adjusted $R^2$

*Pros:* computational advantage, $1+p(p+1)/2$ models; can be applied in high-dimensional when n < p

*Cons:* couldn't find the beset possible models out of all $2^p$ models

**Backward Stepwise Selection**

1. Start from $M_p$ 
2. For k = p, p-1, ..., 1: Fit all k models upon $M_k$ with one predictor out and choose the best as $M_{k-1}$ 
3. Select best model from $M_0, M_1, ..., M_p$ using cross-validated prediction error, AIC, BIC, or adjusted $R^2$

*Cons:* start from full model, so n must larger than p

**Hybrid Approaches**

Start from forward stepwise, after adding each new variable, the method may also remove any variables that no longer provide an improvement in the model fit.

### 3. Choosing the Optimal Model

**Adjustment to the training error**

$$Mallow's \ C_p = \frac{1}{n}(RSS + 2d\hat{\sigma}^2)$$

(d predictors, $\hat{\sigma}$ is estimated using the full model. $C_p$ is an unbiased estimate of test MSE.)

The AIC criterion is defined for a large class of models fit by maximum likelihood. In the case of the model  with Gaussian errors, maximum likelihood and least squares are the same thing. In this case AIC is given by:

*Akaike information criterion:* $AIC = \frac{1}{n\hat{\sigma}^2} (RSS + 2d\hat{\sigma}^2)$

BIC is derived from a Bayesian point of view, but ends up looking similar to $C_p $ (and AIC) as well. For the least squares model with d predictors, the BIC is, up to irrelevant constants, given by

*Bayesian information criterion:* $$BIC = \frac{1}{n\hat{\sigma}^2}(RSS + log(n)d\hat{\sigma}^2)$$

(Since log n > 2 for any n > 7, the BIC statistic generally places a heavier penalty on models with many
variables, and hence results in the selection of smaller models)

$$Adjusted \ R^2 = 1 - \frac{RSS/(n-d-1)}{TSS/(n-1)}$$

**Cross-validation**

*Pros*:

1. directly estimate the test error
2. make fewer assumptions about true model
3. don't need degrees of freedom or estimate the error variance

*One-standard-error rule:* If several models MSEs are close, calculate the standard error of the estimated test MSE for each model size, then select the smallest model for which the estimated test error is within one standard error of the lowest point on the curve.

## Shrinkage Methods

### 1. Ridge Regression

$$\sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p \beta_j^2 = RSS + \lambda \sum_{j=1}^p \beta_j^2 $$

**or**

$$minimize_\beta[\sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2] \ subject \ to \sum_{j=1}^p \beta_j^2 \leq s$$

is minimum, where $\lambda > 0 $ is a tuning parameter, and $\lambda \sum_{j=1}^p \beta_j^2$ is a shrinkage penalty. $||\hat{\beta}||_2$ denotes the $l_2$ from of a vector.

**Scale equivariant**

Different scale can lead to different result, so it is best to apply ridge regression after standardizing the predictors, using the formula:

$$\tilde{x}_{ij} = x_{ij} / \sqrt{\frac{1}{n} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2} = \frac{x_{ij}}{\sigma_j}$$

**Why not Least Squares?**

The least squares estimates will have low bias but may have high variance, whereas ridge regression can still perform well by trading off a small increase in bias for a large decrease in variance. Hence, ridge regression works best in situations where the least squares estimates have high variance.

**Disadvantage: **include all p predictors in the final model

### 2. The Lasso

$$\sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p |\beta_j| = RSS + \lambda \sum_{j=1}^p |\beta_j| $$

**or**

$$minimize_\beta[\sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2] \ subject \ to \sum_{j=1}^p |\beta_j| \leq s$$

**Cons:** yields sparse models, can perform variable selection

### 3. Comparing

Since ridge regression has a circular constraint with no sharp points, this intersection will not generally occur on an axis, and so the ridge regression coefficient estimates will be exclusively non-zero. However, the lasso constraint has corners at each of the axes, and so the ellipse will often intersect the constraint region at an axis. When this occurs, one of the coefficients will equal zero. In higher dimensions, many of the coefficient estimates may equal zero simultaneously.

- When a relatively small number of predictors have substantial coefficients and the remaining are very small or equal to zero, Lasso performs better; if the response is a function of many predictors, Ridge regression will perform better (Number of signal and noise variables)
- Lasso performs variable selection, hence results in models that are easier to interpret.

## Dimension Reduction Methods

Let $Z_1, Z_2, ..., Z_M$ represent M < p linear combinations of original p predictors:

$$Z_m = \sum_{j=1}^p \phi_{jm}X_j$$

We can then fit the linear regression model:

$$y_i = \theta_0 + \sum_{m=1}^M \theta_m z_{im} + \epsilon_i, \ i = 1, ..., n$$

This constraint on the form of the coefficients has the potential to bias the coefficient estimates. However, in situations where p is large relative to n, selecting a value of M p can significantly reduce the variance of the fitted coefficients.

1. Obtain transformed predictors $Z_1, Z_2, ..., Z_M$ 
2. Fit model using M predictors

### 1. Principal Components Regression

- The first principal component is that (normalized) linear combination of the variables with the largest variance.
- The second principal component has largest variance, subject to being uncorrelated with the first.

If the assumption underlying PCR holds, then fitting a least squares model to $Z_1,...,Z_M$ will lead to better results than fitting a least squares model to $X_1,...,X_p$, since most or all of the information in the data that relates to the response is contained in $Z_1,... ,Z_M$, and by estimating only M << p coefficients we can mitigate overfitting.

**Cons:** The response does not supervise the identification of the principal components. There is no guarantee that the directions that best explain the predictors will also be the best directions to use for predicting the response.

### 2. Partial Least Squares

After standardizing the p predictors, PLS computes the first direction $Z_1$ by setting each $\phi_{j1}$ equal to the coefficient from the simple linear regression of Y onto $X_j$. One can show that this coefficient is proportional to the correlation between Y and $X_j$. Hence, in computing $Z_1 = \sum_{j=1}^p \phi_{j1}X_j$, PLS places the highest weight on the variables that are most strongly related to the response.

While the supervised dimension reduction of PLS can reduce bias, it also has the potential to increase variance.

## Considerations in High Dimensions

- When p >= n, least squares cannot be performed, because it will yield a set of coefficient estimates that result in a perfect fit to the data, such that the residuals are zero.

- Unfortunately, the $C_p, AIC, BIC$ approaches are not appropriate in the high-dimensional setting, because estimating $\hat{\sigma}^2$ is problematic. 

- Forward stepwise selection, ridge regression, the lasso, and principal components regression, are particularly useful for performing regression in the high-dimensional setting.

**Curse of dimensionality:** the test error tends to increase as the dimensionality of the problem increases, unless the additional features are truly associated with the response.

**Why?** Because noise features increase the dimensionality of the problem, exacerbating the risk of overfitting without any potential upside in terms of improved test set error.