# Ch3 Linear Regression

*"Essentially, all models are wrong, but some are useful."*   

-- George Box

## Simple Linear Regression

$$Y = \beta_0 + \beta_1 X + \epsilon$$

**Least Squares:** minimize *RSS*

**Residual sum of squares:** $RSS = \sum_{i = 1}^n (y_i - \hat{y_i})^2 = n MSE$ 

> Reference:   
>
> $$E[\frac{RSS}{n-p-1}] = \sigma^2$$

### 1. Coefficient Estimates

$$RSS = (y_1 - \hat{\beta_0} - \hat{\beta_1}x_1)^2 + (y_2 - \hat{\beta_0} - \hat{\beta_1}x_2)^2 + ... + (y_n - \hat{\beta_0} - \hat{\beta_1}x_n)^2$$

To minimize RSS,

$$\hat{\beta_1} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}, SE(\hat{\beta_1}) = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{\sigma^2}{S_{xx}}$$

$$\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{x},\ SE(\hat{\beta_0}) = \sigma^2[\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}]$$

> Reference:   
>
> If $E(x) = \mu$, $Var(x) = \sigma$
>
> $\hat{\mu} = \frac{\sum_{i=1}^n x_i}{n}$, $E(\hat{\mu}) = \mu$, $Var(\hat{\mu}) = \frac{\hat{\mu}}{n}$
>
> $$\hat{\sigma} = \sqrt{\frac{\sum_{i=1}^n (x_i - \hat{\mu})^2}{n-p-1}}, E(\hat{\sigma}) = \sigma$$

In $Y = \beta_0 + \beta_1 X + \epsilon $ :

$$Var(\epsilon) = \sigma^2 , E(\epsilon) = 0 $$

But $\sigma$ is unknown, can be estimated as:

**Residual Standard Error**: $\hat{\sigma} = RSE = \sqrt{\frac{RSS}{n-2}}$ 

### 2. Hypothesis Test

$$H_0: \beta_1 = 0$$

$$H_1: \beta_1 \neq 0$$

$$t-statistic = \frac{\hat{\beta_1} - 0}{SE(\hat{\beta_1})}$$

measures the number of standard deviations that $\hat{\beta_1}$ is away from 0. *(when n > 30, t-distribution is quite similar to the normal distribution)*

**Confidence Intervals**: a range of values such that with 95% probability, the range will contain the true unknow value of the parameter. In other words, the interval will contain true parameter 95% of time.

### 3. Model Accuracy

The RSE provides an absolute measure of lack of fit of the model to the data. But since it is measured in the units of Y, it is not always clear what constitutes a good RSE. The $R^2$ statistic provides an alternative measure of fit.

$$R^2 = \frac{TSS - RSS}{TSS} = 1 - \frac{RSS}{TSS}$$   

$$Total Sum of Squares: TSS = \sum_{i=1}^n (y_i - \hat{y})^2$$

- **TSS:** measures the total variance in the response Y, which is the amount of variability inherent in the response before the regression is performed.
- **RSS:** measures the amount of variability that is left unexplained after performing the regression.
- **TSS - RSS:** measures the amount of variability in the response that is explained (or removed) by performing the regression
- **$R^2$** measures the proportion of variability in Y that can be explained using X

>**Reference:**
>
>$$Cov(x, y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n-1} $$
>
>$$Corr(x, y) = \frac{Cov(x, y)}{\sigma_x \sigma_y} = \frac{S_{xy}}{\sqrt{S_{xx}S_{yy}}} = \hat{\beta_1}\sqrt{\frac{S_{xx}}{S_{yy}}}$$
>
>In OSL 1 predictor:
>
>$$Corr(x, y)^2 = \frac{S_{xy}}{S_{xx}}\frac{S_{xy}}{S_{yy}} = \hat{\beta_1}\frac{S_{xy}}{S_{yy}} = \frac{S_{yy} - RSS}{S_{yy}} = R^2$$ 
>
>*(Page 601 in statistics)*

## Multiple Linear Regression

### 1. Is there a relationship between the response and predictors?

**F test**: $F = \frac{(TSS - RSS) / p}{RSS / (n - p - 1)}$

If a linear model assumptions are correct, then:  

$$E[RSS / (n - p - 1)] = \sigma^2$$

If the estimated model did nothing, then:

$$E[(TSS - RSS) / p] = \sigma^2$$

**Partial F-test:** If we want to test a particular subset of q of the coefficients are 0, we fit a second model uses all predictors except these q and get $RSS_0$ .

**Partial F test**: $F = \frac{(RSS_0 - RSS) / q}{RSS / (n - p - 1)}$

**Why F test if T test is significant?**  

If there are many predictors let's say 100, under 5% confidence level, there will be 5 predictors pass the test by chance.

### 2. Deciding on Important Variables

- **Forward Selection:** begin with null model, add variable which has lowest RSS
- **Backward Selection:** start with all variables, remove the variable with largest p-value
- **Mixed Selection:** start with null, add variables one by one like forward selection; but when new predictors add to model, some p-value will increase, if above a certain value, then remove it.

### 3. Model Fit

$R^2$  will always increase when more variables are added to the model, even if those variables are only weakly associated with the response. 

### 4. Predictions

**Confidence Interval vs Prediction Interval**: confidence interval is an expectation, in many experiments that this interval will include the true value; prediction interval used to quantify the uncertainty Y for a particular X

## Extension of Linear Model

Two most important assumptions:

- **Addictive:** the effect of changes in a predictor $X_j$ on the response Y is independent of the values of the other predictors
- **Linear:** the changes in response Y due to one-unit change in $X_j$ is constant, regardless of the value of $X_j$

### 1. Remove the Additive Assumption

**Interaction doesn't mean collinearity!!!**

There maybe a *synergy effect* or *interaction effect* between predictors.

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X1X2 + \epsilon$$

**Hierarchical Principle:** if we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.

### 2. Non-linear Relationships

Polynomial regression is still linear regression.

## Potential Problems

### 1. Non-linearity

Tool: **Residual plots**

Solution: Try to transform X to $ln(X), \sqrt{X}, X^2$ etc.

### 2. Correlation for Error Terms

$\epsilon_i$ is related to $\epsilon_{i+1}$

If there is correlation, then standard error will tend to underestimate.

### 3. Non-constant Variance of Error Terms (Heteroscedasticity)

funnel shape -> try log Y or sqrt(Y)

### 4. Outliers

**Studentized residuals** = Residual / RSE > 3 Outlier

But be care when remove outliers, because an outlier may instead indicate a deficiency with the model, such as a missing predictor.

### 5. High Leverage Points

High leverage observations tend to have a sizable impact on the estimated regression line.

**Leverage Statistic**: $h_i = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{S_{xx}} $

$$\frac{1}{n} <= h_i <= 1 , \bar{h_i} = \frac{p+1}{n}$$

### 6. Collinearity

Since collinearity reduces the accuracy of the estimates of the regression coefficients, it causes the standard error for $\hat{\beta_j}$ to grow. As a result, in the presence of collinearity, we may fail to reject $H_0: \beta_j = 0$. This means that the power of the hypothesis test--the probability of correctly detecting a non-zero coefficient--is reduced by collinearity.

**Variance Inflation Factor: **   $VIF(\hat{\beta_j}) = \frac{1}{1 - R^2_{X_j|X_{-j}}}$

*$R^2$ is from regression of $ X_j $ onto all of the other predictors. If VIF exceeds 5 or 10, maybe collinearity.*

**Solution:** drop problematic predictors or combine them into a single predictor.

## Linear Regression VS. K-Nearest Neighbors

**Curse of Dimensionality:** The K observations that are nearest to a given test observation $x_0$ may be very far from $x_0$ in p-dimensional space when p is large, leading to a very poor predictions of $f(x_0)$ hence a poor KNN fit.














