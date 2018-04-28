# CrashCourseLM

**Version 1.0.0**

---
*Crash Course on Linear Regression: A Comparison Between Classical and Bayesian Statistics* is a project intended to explain the classical 
and Bayesian approaches applied to a very specific type of model: normal linear regression. It includes a document that explains the
theory and their application to a set of simulated data, and the code written in R and Python used to get these results.

---
## Requirements
Both demos use libraries that may require installment. The Python script `lr.py` uses the following libraries:
- `numpy`
- `math`
- `scipy`
- `statsmodels`
- `matplotlib`
- `pandas`

The R script `lr.R` uses the following libraries:
- `ggplot2`
- `mvtnorm`

---
## Getting Started
Once you get a copy of both scripts you can run them to see how they work, as they are intended to be a demo. The scripts also include
a user-defined function to perform the Bayesina linear regression, called `bayes_lm`, defined in lines 46 and 58 of `lr.R` and `lr.py`,
respectively. The results of the classical regression are obtained using the built-in function `lm` in R and the `OLS` function available
in `statsmodels.formula.api` for Python.

---
## User-Defined Bayesian linear regression function
Both the function written in Python and in R use the same arguments:
- `y` the response vector. In Python: an array of shape (*n*,) (the number of observations). In R: a vector of length *n*.
- `X` the design matrix, which must include the intercept (a column of 1). In Python: an array of shape *(n, p)* (*p* the number of 
columns). In R: a matrix of dimension *(n, p)*.
- `X_tilde` the matrix corresponding to the values. In Python: an array of shape *(m, p)* (*m* the number of new observations), `None`
by default, which generates no estimation of predictions but rather an estimation of `y`. In R: a matrix of dimension *(m, p)*, `NULL`
by default.
- `beta_0` the vector of prior means. In Python: an array of shape *(p,)*, `None` by default, which assumes a vector of 0. In R: a vector
of length *p*, `NULL` by default.
- `Sigma` the prior covariance matrix. In Python: an array of shape *(p, p)*, `None` by default, which assumes a diagonal matrix with
1'000,000 in the diagonal. In R: a matrix of dimension *(p, p)*, `NULL` by default.
- `nu_0` the prior degrees of freedom. In Python: a numeric value, `None` by default, which assumes it to be equal to 1. In R: a numeric
value, `NULL` by default.
- `sigma2_0` the prior scale parameter (of the inverse scaled chi-squared distribution). In Python: a numeric value, `None` by default,
which assumes it to be equal to 1. In R: a numeric value, `NULL` by default.
- `size` the size of the samples generated to make the estimation. The higher this number, the more precise the estimation. In both
Python and R: a numeric value, 10,000 by default.
- `alph` the level of tolerance of the credibility intervals. In both Python and R: a numeric value, 0.05 by default.
- `seed` the seed for the pseudorandom number generators. In both Python and R: a numeric value, 666 by default.

---
## Author
Santiago Lozano Sandino,  
e-mail: <lozsandino@gmail.com>  
LinkedIn user: [lozsandino](https//:www.linkedin.com/in/lozsandino/)


