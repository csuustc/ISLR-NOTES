# Ch4 Classification

## Why not Linear Regression?

- If there are 3 categories in response, which would be 1, 2, 3 in linear regression, then it would be ordinal, because 3>2>1; and they have same interval because 3-2 = 2-1.

- If the response is binary, 0 & 1, it looks reasonable. But the output can exceed [0, 1] which is not interpretable.

> Types of data & measurement scales:  
>
> Nominal -> Ordinal -> Interval -> Ratio

## Logistic Regression

### 1. Model

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1+e^{\beta_0 + \beta_1 X}}$$

To make p(X) from 0 to 1

The odds:  $\frac{p(X)}{1-p(X)} = e^{\beta_0 + \beta_1 X}$

### 2. Estimating

**Likelihood Function:**  $l(\beta_0, \beta_1) = \Pi_{i: y_i = 1} p(x_i) \Pi_{i': y_{i'} = 0} (1 - p(x_{i'}))$

Get $\hat{\beta_0}, \hat{\beta_1}$ to maximize this likelihood.

*Least squares approach is in fact a special case of maximum likelihood.*

### 3. Case-Control

With case-control samples, we can estimate the regression parameters $\beta_j$ accurately, but the constant term $\beta_0$ is incorrect.

If real prevalence rate is $\pi$ , in sample $\tilde{\pi}$ , then the real $\hat{\beta_0^*}$:

$$\hat{\beta_0^*} = \hat{\beta_0} + log(\frac{\pi}{1-\pi}) + log(\frac{\tilde{\pi}}{1-\tilde{\pi}}) $$

Often cases are rare and we take them all; up to five times that number of controls is sufficient.

**Why? -- Diminishing returns in unbalanced binary data**

Sampling more controls than cases reduces the variance of the parameter estimates. But after a ratio of about 5 to 1 the variance reduction flattens out.

### 4. Multiclass Logistic Regression (Multinomial Regression)

$$Pr(Y = k|X) = \frac{e^{\beta_{0k} + \beta_{1k}X_1} + ... + \beta_{pk}X_p}{\sum_{l = 1}^K e^{\beta_{0l} + \beta_{1l}X_1} + ... + \beta_{pl}X_p}$$

> [Wiki Multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression)

## Linear Discriminant Analysis

Why not using logistic regression?

- When the classes are well-separated, logistic regression model are unstable.
- If n is small and the distribution of X is approximately normal in each classes, LDA is more stable than the logistic regression.
- When we have more than 2 response classes, LDA is more popular.

### 1. Bayes' Theorem

The density function of X for an observation from kth class:

$$f_k(x) = Pr(X = x| Y = k)$$

$$Pr(Y = k| X = x) = \frac{\pi_k f_k(x)}{\sum_{l =1}^K \pi_l f_l(x)}$$

### 2. When 1 predictor

Assume $f_k(x)$ is normal or Gaussian:

$$f_k(x) = \frac{1}{\sqrt{2\pi} \sigma_k} exp(-\frac{1}{2\sigma_k^2}(x - \mu_k)^2)$$

*(Here simply assume $\sigma_1 = \sigma_2 = ... = \sigma_k$ , shared variance term across all K classes.)*

$$\hat{\sigma}^2 = \frac{\sum_{k=1}^K \sum_{i: y_i = k} (x_i - \hat{\mu_k})^2}{n-K}$$

**Classifier(discriminant function)**: $\hat{\delta_k (x)} = x \frac{\hat{\mu_k}}{\hat{\sigma}^2} - \frac{\hat{\mu_k^2}}{2\hat{\sigma}^2} + log(\hat{\pi_k})$

### 3. When predictors > 1

- The higher the ratio p/n, the more we expect overfitting
- If true positive rate very low, then null error rate won't be high

**Confusion Matrix**

|                    |    True Null    |  True Non-null  | True Total |
|:------------------:|:---------------:|:---------------:|:----------:|
|   Predicted Null   |  True Neg. (TN) | False Neg. (FN) |     N*     |
| Predicted Non-null | False Pos. (FP) |  True Pos. (TP) |     P*    |
|   Predicted Total  |        N        |        P        |      n     |

- ***FP / N:*** false positive rate, type I error, 1 - specificity
- ***TN / N:*** true negative rate, 1 - type I error, specificity
- ***FN / P:*** false negative rate, type II error
- ***TP / P:*** true positive rate, power, sensitivity, recall
- ***TP / P\*:*** precision, 1 - false discovery proportion
- ***F1 Score:*** 2 * (Precision * Recall) / (Precision + Recall)

**ROC(receiver operating characteristics) & AUC(area under the ROC curve)**

![ROC curve](https://i.stack.imgur.com/PRfzr.png)

### 4. Quadratic Discriminant Analysis

Unlike LDA, QDA assumes that each class has its own covariance matrix. 

LDA is a much flexible classifier than QDA, and so has substantially lower variance. But there is a trade-off, if LDA's assumption that the K class share a common covariance matrix is badly off, then LDA can suffer from high bias. 

Roughly speaking, LDA tends to be a better bet than QDA if there are relatively few training observations and so reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance of the classifier is not a major concern, or if the assumption of a common covariance matrix for the K classes is clearly untenable.

## Comparison

LDA assumes that the observations are drawn from a Gaussian distribution with a common covariance matrix in each class, and so can provide some improvements over logistic regression when this assumption approximately holds. Conversely, logistic regression can
outperform LDA if these Gaussian assumptions are not met.

When the decision boundary is highly non-linear, use KNN. On the other hand, KNN does not tell us which predictors are important.

When the true decision boundaries are linear, then the LDA and logistic regression approaches will tend to perform well. When the boundaries are moderately non-linear, QDA may give better results. Finally, for much more complicated decision boundaries, a non-parametric approach such as KNN can be superior.







