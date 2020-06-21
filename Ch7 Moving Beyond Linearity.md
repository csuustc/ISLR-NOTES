# Ch7 Moving Beyond Linearity

## Polynomial Regression

$$y_i = \beta_0 + \beta_1x_i + \beta_2x_i^2 + \beta_3x_i^3 + ... + \beta_dx_i^d + \epsilon_i$$

Generally, it's unusual to us d greater than 3 or 4 because for large values of d, the polynomial curve can become overly flexible and can take on some very strange shapes.

**Cons:**

Tail behavior really bad

- data is less at tail
- polynomial's tail more flexible

## Step Functions

$$y_i = \beta_0 + \beta_1C_1(x_i) + \beta_2C_2(x_i) + \beta_3C_3(x_i) + ... + \beta_KC_K(x_i) + \epsilon_i$$

Cons: constant within each region, no trends

## Regression Splines

The general definition of a degree-d spline is that it is a piecewise degree-d polynomial, with continuity in derivatives up to degree d − 1at each knot.

### 1. Linear Splines

Degree 1 spline only have continuity at knots.

Let's say we have K knots at $\xi_1, \xi_2, ... \xi_K$, the model is:

$$y_i = \beta_0 + \beta_1b_1(x_i) + \beta_2b_2(x_i) + \beta_3b_3(x_i) + ... + \beta_{K+1}b_{K+1}(x_i) + \epsilon_i$$

where $b_k$ are basis functions

$$b_1(x_i) = x_i, b_2(x_i) = (x_i - \xi_1)_+, ..., b_{K+1}(x_i) = (x_i - \xi_K)_+$$

Linear spline is a piecewise linear polynomial continuous at each knot.

### 2. Cubic Splines

Degree 3 spline impose 3 constraints: continuity, continuity of 1st derivatives, continuity of 2nd derivatives. (K + 4 degree freedom)

$$y_i = \beta_0 + \beta_1b_1(x_i) + \beta_2b_2(x_i) + \beta_3b_3(x_i) + ... + \beta_{K+3}b_{K+3}(x_i) + \epsilon_i$$

where $b_k$ are truncated power basis functions

$$b_1(x_i) = x_i, b_2(x_i) = x_i^2, b_3(x_i) = x_i^3,..., b_{K+3}(x_i) = (x_i - \xi_K)_+^3$$

**Natural Spline**

Unfortunately, splines can have high variance at the outer range of the predictors. Natural spline is required to be linear at the boundary (in the region where X is smaller than the smallest knot, or larger than the largest knot). Add 2 * 2 extra constraints. 

### 3. Knots Placement

1. Where to place knots?
   - place them at appropriate quantiles of the observed X
2. How many K?
   - Cross-validation

### 4. Comparison to Polynomial Regression

Regression splines often give superior results to polynomial regression. Because polynomial regression has to use a high degree to produce flexible fits, however splines introduce flexibility by increasing the number of knots but keeping the degree fixed. Generally, this approach produces more stable estimates. Splines also allow us to place more knots, and hence flexibility, over regions where the function f seems to be changing rapidly, and fewer knots where f appears more stable. 

## Smoothing Splines

Regression splines use least squares to minimize RSS and get the coefficients. Generally, when we minimize RSS using g(x), it could make RSS 0 by choosing g interpolates all the y, which will overfit data. Here we want to make a trade-off between RSS and smooth of g(x):

$$minimize_g \sum_{i=1}^n (y_i - g(x_i))^2 + \lambda \int g''(t)^2 dt$$

Where $\lambda$ is a nonnegative tuning parameter. RSS is a loss function and $\lambda \int g''(t)^2 dt$ is a penalty term, which is a measure of g's roughness.

* When $\lambda = 0$, g will be very jumpy and will exactly interpolate the training observations
* When $\lambda \to \infin$, g will be perfectly smooth—it will just be a straight line that passes as closely as possible to the training points.

Actually, the g(x) is a **natural cubic spline** with knots at $x_1, x_2, ..., x_n$ 

### Choosing Smoothing Parameter

When $\lambda$ increases from 0 to infinity, the **effective degrees of freedom**, which we write $df_\lambda$ decrease from n to 2.

$$\hat{g}_\lambda = S_\lambda y, \ df_\lambda = \sum_{i=1}^n \{S_\lambda\}_{ii}$$

How to decide $\lambda$?

- Cross-validation
- In leave-one cross-validation, $RSS_{cv}(\lambda) = \sum_{i=1}^n [\frac{y_i - \hat{g}_\lambda(x_i)}{1-\{S_\lambda\}_{ii}}]^2$

*(Smoothing splines avoid the knot-selection issue, leaving a single $\lambda$ to be chosen.)*

## Local Regression

1. Gather s = k/n training points whose $x_i$ are closest to $x_0$ 
2. Assign a weight $K_{i0} = K(x_i, x_0)$ to each point in this neighborhood
3. Weighted least squares regression, minimize $\sum_{i=1}^nK_{i0}(y_i - \beta_0 - \beta_1x_i)^2$
4. $\hat{f}(x_0) = \hat{\beta}_0 + \hat{\beta}_1x_0$

Local regression is sometimes referred to as a memory-based procedure, because like nearest-neighbors, we need all the training data each time we wish to compute a prediction. The smaller the value of s,the more local and wiggly will be our fit; alternatively, a very large value of s will lead to a global fit to the data using all of the training observations.

Local regression also generalizes very naturally when we want to fit models that are local in a pair of 2  variables, rather than one. We can simply use two-dimensional neighborhoods, and fit bivariate linear regression models using the observations that are near each target point in two-dimensional space. However, local regression can perform poorly if p is much larger than about 3 or 4 because there will generally be very few training observations near that.

## Generalized Additive Models

### 1. GAMs for Regression Problems

$$y_i = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + ... + f_p(x_{ip}) + \epsilon_i$$

*(It is called an additive model because we calculate a separate $f_j$ for each $X_j$ and then add together all of their contributions.)*

GAMs provide a useful compromise between linear and fully nonparametric models.

**Pros:**

* Can model non-linear relationships on each variable individually
* More accurate
* Can be easily interpreted because individually
* Can be summarized via degrees of freedom

**Cons:**

Important interactions can be missed. But we can add low-dimensional interaction functions into the model.

### 2. GAMs for Classification Problems

$$log(\frac{p(X)}{1-p(X)}) = \beta_0 + f_1(X_{1}) + f_2(X_{2}) + ... + f_p(X_{p})$$

