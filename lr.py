import numpy as np
import math as mt
import scipy as sp
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd

# Data simulation.
np.random.seed(666)
n = 30
X = np.column_stack((np.ones(n), np.random.normal(15, 5, n)))
b = np.array((5, 30)).T
p = X.shape[1]
sigma = 30
y = X @ b + np.array((np.random.normal(0, sigma, n))).T
x_tilde = np.arange(mt.floor(min(X[:, 1])), mt.ceil(max( X[:, 1])) + 1, 1)
X_tilde = np.column_stack((np.ones(len(x_tilde)), x_tilde))

# Classical linear regression.
fit = smf.OLS(y, X).fit()
print(fit.summary())
dir(fit)
y_hat = fit.predict(X_tilde)
sigma_hat = np.sqrt(fit.mse_resid)
alph = 0.05

# Graphical representation.
def linreg(y, X, y_hat, sigma_hat, X_tilde):
    invXtX = np.linalg.inv(X.T @ X)
    np.sqrt(np.diag(X_tilde @ invXtX @ X_tilde.T))
    plt.scatter(X[:, 1], y, c = '#000000', alpha = 0.5)
    plt.plot(X_tilde[:, 1], y_hat, c = '#FF0000')
    plt.plot(X_tilde[:, 1], y_hat + 
             sp.stats.t.ppf(0.975, n - p) * sigma_hat * np.sqrt(np.diag(X_tilde @ invXtX @ X_tilde.T)), 
             c = "#00FF00")
    plt.plot(X_tilde[:, 1], y_hat - 
             sp.stats.t.ppf(0.975, n - p) * sigma_hat * np.sqrt(np.diag(X_tilde @ invXtX @ X_tilde.T)), 
             c = "#00FF00")
    plt.plot(X_tilde[:, 1], y_hat + 
             sp.stats.t.ppf(0.975, n - p) * sigma_hat * np.sqrt(1 + np.diag(X_tilde @ invXtX @ X_tilde.T)), 
             c = "#0000FF")
    plt.plot(X_tilde[:, 1], y_hat - 
             sp.stats.t.ppf(0.975, n - p) * sigma_hat * np.sqrt(1 + np.diag(X_tilde @ invXtX @ X_tilde.T)), 
             c = "#0000FF")
    plt.xlabel("x")
    plt.ylabel("y")

linreg(y, X, y_hat, sigma_hat, X_tilde)

# Sample from a multivariate t distribution.
def random_mv_t(df, Sigma, n):
    p = Sigma.shape[1]
    den = np.sqrt(np.random.chisquare(df, n)/df)
    num = np.random.multivariate_normal(np.zeros(p), Sigma, n)
    return num/den[:, None]

# Bayesian linear regression.
def bayes_lm(y, X, X_tilde = None,
             beta_0 = None, Sigma = None, nu_0 = None, sigma2_0 = None,
             size = 10000, alph = 0.05, seed = 666):
    
    # Default values.
    np.random.seed(seed)
    n = len(y)
    p = X.shape[1]
    if beta_0 is None:
        beta_0 = np.zeros(p).T
    if Sigma is None:
        Sigma = 1000000 * np.diag(np.ones(p))
    if nu_0 is None:
        nu_0 = 1
    if sigma2_0 is None:
        sigma2_0 = 1
    
    # Fixed values and auxiliar variables.
    V = X.T @ X
    beta_hat = np.linalg.inv(V) @ X.T @ y
    Sigma_n = np.linalg.inv(np.linalg.inv(Sigma) + V)
    eig_Sigma_n = np.linalg.eigh(Sigma_n)
    eig_Sigma_n = eig_Sigma_n[1] @ np.diag(np.sqrt(eig_Sigma_n[0]))
    beta_n = Sigma_n @ (X.T @ y + Sigma @ beta_0)
    S2 = sum((y - X @ beta_hat)**2)/(n - p)
    
    # Simulation of sigma2 and beta.
    sigma2_sample = (nu_0 * sigma2_0 + (n - p) * S2)/np.random.chisquare(n - p, size)
    beta_sample = np.random.normal(0, 1, size * p)
    beta_sample.shape = (p, size)
    beta_sample = eig_Sigma_n @ (beta_sample * np.sqrt(sigma2_sample)) + beta_n[:, None]
    
    # Prediction.
    if X_tilde is None:
        y_hat = X @ beta_sample
    else:
        m = X_tilde.shape[0]
        sim = random_mv_t(n - p, np.diag(np.ones(m)), size)
        delta = X_tilde @ beta_hat
        y_hat = sim.T * np.sqrt(S2) + delta[:, None]
    
    # Results.
    results = np.column_stack((beta_sample.mean(axis = 1), 
                               np.percentile(beta_sample, axis = 1, q = alph/2 * 100), 
                               np.percentile(beta_sample, axis = 1, q = 100 - alph/2 * 100)))
    sig = np.array((np.mean(sigma2_sample), 
                    np.percentile(sigma2_sample, q = alph/2 * 100), 
                    np.percentile(sigma2_sample, q = 100 - alph/2 * 100)))
    results = np.vstack((results, sig))
    param = ["beta_" + str(i) for i in range(p)]
    param.append("sigma2")
    results = pd.DataFrame({"mean": results[:, 0],
                           "lower": results[:, 1],
                           "upper": results[:, 2]})
    results.index = param
    y_hat = np.column_stack((y_hat.mean(axis = 1), 
                             np.percentile(y_hat, axis = 1, q = alph/2 * 100),
                             np.percentile(y_hat, axis = 1, q = 100 - alph/2 * 100)))
    y_hat = pd.DataFrame({"mean": y_hat[:, 0],
                          "lower": y_hat[:, 1],
                          "upper": y_hat[:, 2]})
    return {"results": results,
            "beta_sample": beta_sample.T,
            "sigma2_sample": sigma2_sample,
            "y_hat": y_hat}

fit1 = bayes_lm(y, X)    
fit2 = bayes_lm(y, X, X_tilde)
fit2["results"]

def bayes_linreg(y, x, y_hat, y_hat2, x_tilde):
    plt.scatter(x, y, c = "#000000", alpha = 0.5)
    plt.plot(x_tilde, y_hat["mean"], c = "#FF0000")
    plt.plot(x, y_hat2["lower"], c = "#00FF00")
    plt.plot(x, y_hat2["upper"], c = "#00FF00")
    plt.plot(x_tilde, y_hat["lower"], c = "#0000FF")
    plt.plot(x_tilde, y_hat["upper"], c = "#0000FF")
    plt.xlabel("x")
    plt.ylabel("y")

y_hat = fit2["y_hat"]
y_hat2 = fit1["y_hat"]
x = X[:,1]
x_tilde = X_tilde[:,1]

bayes_linreg(y, x, y_hat, y_hat2, x_tilde)
