# Ch9 Support Vector Machines

Here we approach the two-class classification problem in a direct way: 

*We try and find a plane that separates the classes in feature space.*

If we cannot, we get creative in two ways: 

* We soften what we mean by “separates”, and
* We enrich and enlarge the feature space so that separation is possible.

## Maximal Margin Classifier

### 1. Separating Hyperplane

In a p-dimensional space, a hyperplane is a flat affine subspace of dimension p − 1.

$$\beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p = 0$$

The vector $\vec{\beta} = (\beta_1, \beta_2, ... , \beta_p)$ is called the normal vector — it points in a direction orthogonal to the surface of a hyperplane.

In general, if our data can be perfectly separated using a hyperplane, then there will in fact exist an infinite number of such hyperplanes.

A natural choice is the **maximal margin hyperplane** (also known as the **optimal separating hyperplane**), which is the separating hyperplane that is farthest from the training observations, that makes the biggest gap or margin between the two classes.

**Support Vector:** the maximal margin hyperplane depends directly on only a small subset of the observations 

### 2. Construction of the Maximal Margin Classifier

$$Maximize_{\beta_0, \beta_1, ..., \beta_p, M} M \ subject \ to \ \sum_{j=1}^p \beta_j^2 = 1$$

$$y_i(\beta_0 + \beta_1X_{i1} + \beta_2X_{i2} + ... + \beta_pX_{ip}) \geq M \  \forall i = 1, 2, ..., n$$

(M is the margin distance)

## Support Vector Classifiers

**The support vector classifier**, sometimes called a **soft margin classifier**, maximizes a soft margin. We allow some observations to be on the incorrect side of the margin, or even the incorrect side of the hyperplane. (The margin is soft because it can be violated by some of the training observations.) 

$$Maximize_{\beta_0, \beta_1, ..., \beta_p, \epsilon_1, ..., \epsilon_n, M} M \ subject \ to \ \sum_{j=1}^p \beta_j^2 = 1$$

$$y_i(\beta_0 + \beta_1X_{i1} + \beta_2X_{i2} + ... + \beta_pX_{ip}) \geq M(1-\epsilon_i) \  \forall i = 1, 2, ..., n$$

$$\epsilon_i \geq 0, \sum_{i=1}^n \epsilon_i \leq C$$

Where C is a nonnegative tuning parameter. If $\epsilon_i > 0$ then the i-th observation is on the wrong side of the margin, and we say that the i-th observation has violated the margin. If $\epsilon_i > 1$ then it is on the wrong side of the hyperplane.

C bounds the sum of the i’s, and so it determines the number and severity of the violations to the margin (and to the hyperplane) that we will tolerate.

Observations that lie directly on the margin, or on the wrong side of the margin for their class, are known as **support vectors**. These observations do affect the support vector classifier.

## Support Vector Machines

The support vector machine (SVM) is an extension of the support vector classifier that results from enlarging the feature space in a specific way, using kernels. 

Why not polynomials?

Polynomials (especially high-dimensional ones) get wild rather fast. Kernel is a more elegant and controlled way to introduce nonlinearities in support-vector classifiers.

The linear support vector classifier can be represented as:

$$f(x) = \beta_0 + \sum_{i=1}^n \alpha_i\langle x, x_i \rangle$$

If a training observation is not a support vector, then its $\alpha_i$ equals zero.

So if S is the collection of indices of support points,

$$f(x) = \beta_0 + \sum_{i \in S}^n \alpha_i\langle x, x_i \rangle$$

$$f(x) = \beta_0 + \sum_{i \in S}^n \alpha_iK(x, xi)$$

**Linear kernel:** $K(x_i, x_{i'}) = \sum_{j=1}^p x_{ij}x_{i'j}$

**Polynomial kernel of degree d: **$K(x_i, x_{i'}) = (1 + \sum_{j=1}^p x_{ij}x_{i'j})^d$

**Radial kernel:** $K(x_i, x_{i'}) = exp(-\gamma \sum_{j=1}^p (x_{ij} - x_{i'j})^2)$

## SVM with More than Two Classes

### 1. One-Versus-One Classification

A one-versus-one or all-pairs approach constructs $C_K^2$ SVMs, and assign the test to the class that wins the most pairwise competitions.

### 2. One-Versus-All Classification

Fit K SUMs each time comparing one of the K classes to the remaining K − 1 classes. Classify test to the class for which f(x) us largest.

**Which to choose?** If K is not too large, use OVO.

## Relationship to Logistic Regression

* When classes are (nearly) separable, SVM does better than LR. So does LDA.
* When not, LR (with ridge penalty) and SVM very similar. But LR can provide probability, which is interpretable easily.
* If you wish to estimate probabilities, LR is the choice.
* For nonlinear boundaries, kernel SVMs are popular. Can use kernels with LR and LDA as well, but computations are more expensive.
* SVM uses all features when fitting, so couldn't get feature selection.



