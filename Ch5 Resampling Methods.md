# Ch5 Resampling Methods

**Why Resampling?**

- Test set is often not available. Otherwise we just use a large designated test set.
- But we need a method to predict the test set error, the standard deviation and bias, etc. 
   1. Mathematical adjustment like Cp statistic, AIC, BIC
   2. Holding out a subset of the training observations

## Cross-Validation

### 1. The Validation Set Approach

Usually randomly split the whole data set into 50% training data and 50% test data, which is very easy but has potential drawbacks:

1. The validation estimate of the test error rate can be highly variable. It depends on which observations are included in the training set.
2. Only a subset of observations are included in the training set, which performs worse when trained on fewer observations. This suggests the prediction error rate maybe overestimated.

### 2. Leave-One-Out Cross-Validation (LOOCV)

Fit n times using n-1 observations, but maybe too complex to compute.

**Estimated MSE:** $CV_n = \frac{1}{n} \sum_{i=1}^n MSE_i$ 

With least squares linear or polynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single model fit:

$$CV_n = \frac{1}{n} \sum_{i=1}^n (\frac{y_i - \hat{y_i}}{1-h_i})^2$$

### 3. K-Fold Cross-Validation

Split the whole dataset into K groups, then fit k times: (usually k = 5 or 10)

$CV_n = \frac{1}{k} \sum_{i=1}^k MSE_i$ 

### 4. Bias-Variance Trade-Off for K-Ford Cross-Validation

* LOOCV has more observations in training set so that get approximately unbiased estimates of the test error.
* But LOOCV has higher variance because most observations are overlapped. 

### 5. Cross-Validation on Classification Problems

$CV_n = \frac{1}{n} \sum_{i=1}^n Err_i$ 

### 6. Attention in Predictors Selection

CV first, predictors selection after. Because the selection procedure has already seen the labels of the training data, and made use of them. This is a form of training and must be included in the validation process.

## The Bootstrap

- For real data, we cannot generate new samples from the original population usually.
- However, the bootstrap approach allows us to use a computer to mimic the process of obtaining new data sets, so that we can estimate the variability of our estimate without generating additional samples.
- Rather than repeatedly obtaining independent data sets from the population, we instead obtain distinct data sets by repeatedly sampling observations from the original data set with replacement.

For example, we get B simulated groups and corresponding B groups estimates:

$$SE_B(\hat{\alpha}) = \sqrt{\frac{1}{B-1} \sum_{r=1}^B (\hat{\alpha}^{*r} - \bar{\hat{\alpha}}^*)^2}$$

**How to apply bootstrap in timeseries data?**

We can instead create blocks of consecutive observations, and sample those with replacements. Then we paste together sampled blocks to obtain a bootstrap dataset.

**Confidence intervals for parameter:**

For B simulated groups, we get B estimate for parameters, so that we can get a percentage interval which is called Bootstrap Percentile confidence interval. It is the simplest method (among many approaches) for obtaining a confidence interval from the bootstrap.

**Why cannot estimate prediction error?**

- In Cross-Validation, we have the test set which there is no overlap, and that's the crucial point.
- Bootstrap has about 2/3 overlap in each bootstrap sample and will cause underestimate the true prediction error
- But we can use points excluding the bootstrap points as validation
