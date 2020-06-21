# Ch2 Statistical Learning

- Input - *predictors, independent variables, features*

- Output - *response, dependent variables*

**Regression Function**

If population is known, then:

$$f(x) = E(Y | X = x)$$

## Why Estimate f?

### 1. Prediction

$$Y = f(x) + \epsilon$$

$$\hat{Y} = \hat{f(x)}$$

$$E[(Y - \hat{f(x)})^2 | X = x] = [f(x) - \hat{f(x)}]^2 + Var(\epsilon)$$

* reducible error: $\hat{f}$ will not be a perfect estimate for f, and this inaccuracy will introduce some error, and can be improved by using the most appropriate technique to estimate f.
* irreducible error: even if we got perfect estimate for f, $\epsilon$ still cannot be predicted using ***X***. No matter how well we estimate f, we cannot reduce the error introduced by $\epsilon$ .
	* why irreducible? - there are some predictors that are useful but we didn't measure them, which means f doesn't use them; and some of these predictor maybe couldn't measure at all.

### 2. Inference

Understand the relationship between ***X*** and ***Y***.

## How Estimate f?

We don't know the population so we cannot compute $f(x) = E(Y | X = x)$ , but can choose a near neighbor set to estimate.

### 1. Parametric Methods

Two step: Model Selection -> Parameter Estimate

**Disadvantage**: We don't know the true f.   

- But we can choose flexible models that can fit many different possible functional forms for *f*. 
- But the more complex models can lead to overfitting, and require estimating a greater number of parameters.

### 2. Non-parametric Methods

Can avoid the assumption of a particular functional form for _f_

**Disadvantage**: need a very large number of observations  

### 3. Trade-Off between Prediction Accuracy and Model Interpretability

Restrictive models are much more interpretable.

## Fit Quality

**Mean Squared Error:** $MSE = \frac{1}{n} \sum_{i=1}^n(y_i - \hat{f(x_i)})^2$

- Training MSE can decrease lower and lower -> overfitting
- But Test MSE have a minimum and always bigger than $Var(\epsilon)$

**Overfitting**: a small training MSE & a large test MSE  

(*Working too hard to find patterns in the training data, and may be picking up some patterns that are just caused by random chance rather than by true properties of the unknown function f*)

### Bias-Variance Trade-Off

$$E[(y_0 - \hat{f(x_0)})^2] = [\hat{f(x_o)} - E(\hat{f(x_0))}]^2 + [E(\hat{f(x_0)} - f(x_0))]^2 + \epsilon^2 = Var(\hat{f(x_0)}) + [Bias(\hat{f(x_0)})]^2 + Var(\epsilon)$$

- **Variance** refers to the amount by which $\hat{f}$ would change if we estimated it using a different training data set. In general, more flexible statistical methods have higher variance.
- **Bias** refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated , by a much simpler model. Generally, more flexible methods result in less bias.

> **Reference:**  
>
> **Point Estimator Bias:** $E(\hat{x}) - x$ 
>
> $$E(\hat{x} - x)^2 = E(\hat{x} - E(\hat{x}) + E(\hat{x}) - x)^2 = E(\hat{x} - E(\hat{x}))^2 + E(E(\hat{x}) - x)^2 = Var(\hat{x}) + [Bias(\hat{x})]^2$$
>
> [MSE and Bias-Variance decomposition](https://towardsdatascience.com/mse-and-bias-variance-decomposition-77449dd2ff55)  
>
> [The Bias-Variance Tradeoff](https://towardsdatascience.com/the-bias-variance-tradeoff-8818f41e39e9)
>
> > ![operation](https://miro.medium.com/max/1396/1*4mIiXP_K3kfhr9O_irLjfQ.png)

### Bayes Classifier

Assign to class *j* when following $Pr$ is largest   

$$Pr(Y = j | X = x_0)$$

### K-Nearest Neighbors

In Theory we should always predict qualitative responses using the Bayes Classifier. But for real data, it's impossible because we don't know the real conditional distribution. So attempt to estimate the conditional distribution.

$$Pr(Y = j | X = x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i = j)$$




















