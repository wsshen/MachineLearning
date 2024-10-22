library(MASS)     # mvrnorm() for multivariate normal
set.seed(11)        # random seed set for reproducibility
n <- 60              # sampling 60 each from 4 separate distributions
m1 <- c(1.5, 1.5)   # upper right x, y means
S1 <- matrix(c(0.3, 0.05, 0.05, 0.3), ncol = 2) # variance of c1
clus1 <- MASS::mvrnorm(n = n, mu = m1, Sigma = S1)
m2 <- c(1.5, -1.5)
S2 <- matrix(c(0.5, -0.08, -0.08, 0.2), ncol = 2)

clus2 <- MASS::mvrnorm(n = n, mu = m2, Sigma = S2)
m3 <- c(-1.5, 1.5)  # upper left
S3 <- matrix(c(0.1, 0.03, 0.03, 0.1), ncol = 2)
clus3 <- MASS::mvrnorm(n = n, mu = m3, Sigma = S3)
m4 <- c(-1.5, -1.5) # lower left
S4 <- matrix(c(0.8, 0.50, 0.50, 0.8), ncol = 2)
clus4 <- MASS::mvrnorm(n = n, mu = m4, Sigma = S4)
datc <- rbind(clus1, clus2, clus3, clus4) # 240 observations altogether
# run the CRP Gibbs function in Appendix B.
alpha <- 0.01
mu0 <- matrix(rep(0, 2), ncol = 2, byrow = TRUE)
sigma0 <- diag(2) * 3^2
sigma_y <- diag(2) * 1
c_init <- rep(1, nrow(datc))
z <- c_init
n_k <- as.vector(table(z))
Nclust <- length(n_k)
N <- nrow(datc)
n_k
Nclust
N
for(n in 1:N) 
  c_i <- z[n]
n_k[c_i] <- n_k[c_i] - 1 
if( n_k[c_i] == 0 )
{
  n_k[c_i] <- n_k[Nclust]  # last cluster to replace this empty cluster
  loc_z <- ( z == Nclust ) # who are in the last cluster?
  z[loc_z] <- c_i          # move them up to fill just emptied cluster
  n_k <- n_k[ -Nclust ]    # take out the last cluster, now empty
  Nclust <- Nclust - 1     # decrease total number of clusters by 1
}


import numpy as np
from numpy.random import multivariate_normal

np.random.seed(11)  # random seed set for reproducibility
n = 60  # sampling 60 each from 4 separate distributions
m1 = np.array([1.5, 1.5])  # upper right x, y means
S1 = np.array([[0.3, 0.05], [0.05, 0.3]])  # variance of c1
# sampling n from each cluster as per its mean mu and variance Sigma
clus1 = multivariate_normal(mean=m1, cov=S1, size=n)
m2 = np.array([1.5, -1.5])  # lower right
S2 = np.array([[0.5, -0.08], [-0.08, 0.2]])
clus2 = multivariate_normal(mean=m2, cov=S2, size=n)
m3 = np.array([-1.5, 1.5])  # upper left
S3 = np.array([[0.1, 0.03], [0.03, 0.1]])
clus3 = multivariate_normal(mean=m3, cov=S3, size=n)
m4 = np.array([-1.5, -1.5])  # lower left
S4 = np.array([[0.8, 0.5], [0.5, 0.8]])
clus4 = multivariate_normal(mean=m4, cov=S4, size=n)
datc = np.vstack((clus1, clus2, clus3, clus4))  # 240 observations altogether

# run the CRP Gibbs function in Appendix B.
alpha = 0.01
mu0 = np.zeros((1, 2))
sigma0 = np.diag([3**2, 3**2])
sigma_y = np.diag([1, 1])
c_init = np.ones(datc.shape[0])
z = c_init.copy()
n_k = np.bincount(z.astype(int))

Nclust = len(n_k)
print(n_k,Nclust)
for n in range(1):
    c_i = int(z[n])
    
    n_k[c_i] -= 1
    if n_k[c_i] == 0:
        n_k[c_i] = n_k[Nclust - 1]  # last cluster to replace this empty cluster
        loc_z = (z == Nclust)  # who are in the last cluster?
        z[loc_z] = c_i  # move them up to fill just emptied cluster
        n_k = np.delete(n_k, Nclust - 1)  # take out the last cluster, now empty
        Nclust -= 1  # decrease total number of clusters by 1



