library(ggplot2) # Graphics.
library(mvtnorm) # Multivariate normal and multivariate t.

# Data simulation.
set.seed(69)
n <- 30
X <- cbind(1, rnorm(n, 15, 5))
p <- ncol(X)
b <- c(3, 30)
sigma <- 30
y <- X %*% b + rnorm(n, 0, sigma)
train_data <- as.data.frame(cbind(y, X[, 2]))
names(train_data) <- c("y", "x")
X_tilde <- as.data.frame(cbind(y = NA, 
                               x = seq(floor(min(X[, 2])), ceiling(max(X[, 2])))))

# Classical linear regression.
fit <- lm(y ~ x, train_data)
sum.fit <- summary(fit)
sum.fit
names(fit)
y_hat_ci <- predict(fit, X_tilde, interval = "confidence")
y_hat_pi <- predict(fit, X_tilde, interval = "prediction")

# Graphical representation of the classical linear regression.
ggplot() + geom_point(aes(y = y, x = X[, 2]), alpha = 0.5) +
  geom_line(aes(y = y_hat_ci[, "fit"], 
                x = X_tilde[, 2]), 
            colour = "#FF0000") +
  geom_line(aes(y = y_hat_ci[, "lwr"],
                x = X_tilde[, 2]), 
            colour = "#00FF00") +
  geom_line(aes(y = y_hat_ci[, "upr"],
                x = X_tilde[, 2]), 
            colour = "#00FF00") +
  geom_line(aes(y = y_hat_pi[, "lwr"],
                x = X_tilde[, 2]), 
            colour = "#0000FF") +
  geom_line(aes(y = y_hat_pi[, "upr"],
                x = X_tilde[, 2]), 
            colour = "#0000FF") +
  scale_x_continuous("x", breaks = seq(5, 30, 5)) +
  scale_y_continuous("y", breaks = seq(0, 1000, 200), limits = c(0, 1000))

# Bayesian linear regression.
bayes_lm <- function(y, X, X_tilde = NULL,
                     beta_0 = NULL, Sigma = NULL, nu_0 = NULL, sigma2_0 = NULL, 
                     size = 10000, alph = 0.05, seed = 666) {
  
  # Default values.
  set.seed(seed)
  p <- ncol(X)
  n <- length(y)
  if (is.null(beta_0)) {
    beta_0 <- rep(0, p)
  }
  if (is.null(Sigma)) {
    Sigma <- diag(rep(100000, p))
  }
  if (is.null(nu_0)) {
    nu_0 <- 1
  }
  if (is.null(sigma2_0)) {
    sigma2_0 <- 1
  }
  
  # Fixed values and auxiliar variables.
  V <- t(X) %*% X
  beta_hat <- solve(V) %*% t(X) %*% y
  Sigma_n <- solve(solve(Sigma) + V)
  eig_Sigma_n <- eigen(Sigma_n)
  eig_Sigma_n <- eig_Sigma_n$vectors %*% diag(sqrt(eig_Sigma_n$values)) # spectral decomposition.
  beta_n <- Sigma_n %*% (t(X) %*% y + Sigma %*% beta_0)
  S2 <- sum((y - X %*% beta_hat)^2)/(n - p)
  
  # Simulation of sigma2 and beta.
  sigma2_sample <- (nu_0 * sigma2_0 + (n - p) * S2)/rchisq(size, n - p)
  beta_sample <- t(matrix(rnorm(p * size), ncol = p) * sqrt(sigma2_sample))
  beta_sample <- eig_Sigma_n %*% beta_sample + as.vector(beta_n)
  
  # Prediction.
  if (is.null(X_tilde)) {
    y_hat <- X %*% beta_sample
  } else {
    m <- nrow(X_tilde)
    y_hat <- t(rmvt(size, sigma = diag(rep(1, m)), df = n - p) * sqrt(S2)) +
      c(X_tilde %*% beta_hat)
  }
  
  # Results.
  results <- cbind(rowMeans(beta_sample), 
                   t(apply(beta_sample, 1, quantile, probs = c(alph/2, 1 - alph/2))))
  results <- rbind(results, c(mean(sigma2_sample), 
                              quantile(sigma2_sample, probs = c(alph/2, 1 - alph/2))))
  results <- as.data.frame(results)
  results <- cbind(c(paste0("beta_", 0:(p - 1)), "sigma2"), results)
  names(results) <- c("param", "mean", "lower", "upper")
  y_hat <- cbind(seq(1, nrow(y_hat)), rowMeans(y_hat),
                  t(apply(y_hat, 1, quantile, probs = c(alph/2, 1 - alph/2))))
  y_hat <- as.data.frame(y_hat)
  names(y_hat) <- c("obs", "mean", "lower", "upper")
  return(list(results = results,
              beta_sample = beta_sample,
              sigma2_sample = sigma2_sample,
              y_hat = y_hat))
}

X_tilde <- cbind(1, x = seq(floor(min(X[, 2])), ceiling(max(X[, 2]))))
fit1 <- bayes_lm(y, X)
fit2 <- bayes_lm(y, X, X_tilde)
fit2$results

# Graphical representation of the Bayesian linear regression.
ggplot() + geom_point(aes(y = y, x = X[, 2]), alpha = 0.5) +
  geom_line(aes(y = fit2$y_hat[, "mean"], 
                x = X_tilde[, 2]), 
            colour = "#FF0000") +
  geom_line(aes(y = fit1$y_hat[, "lower"],
                x = X[, 2]), 
            colour = "#00FF00") +
  geom_line(aes(y = fit1$y_hat[, "upper"],
                x = X[, 2]), 
            colour = "#00FF00") +
  geom_line(aes(y = fit2$y_hat[, "lower"],
                x = X_tilde[, 2]), 
            colour = "#0000FF") +
  geom_line(aes(y = fit2$y_hat[, "upper"],
                x = X_tilde[, 2]), 
            colour = "#0000FF") +
  scale_x_continuous("x", breaks = seq(5, 30, 5)) +
  scale_y_continuous("y", breaks = seq(0, 1000, 200), limits = c(0, 1000))
