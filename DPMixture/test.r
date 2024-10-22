library("MASS")     # mvrnorm() for multivariate normal
set.seed(11)        # random seed set for reproducibility
n <- 60              # sampling 60 each from 4 separate distributions
m1 <- c(1.5, 1.5)   # upper right x, y means
S1 <- matrix(c(0.3, 0.05, 0.05, 0.3), ncol = 2) # variance of c1
 # sampling n from each cluster as per its mean mu and variance Sigma
clus1 <- mvrnorm(n = n, mu = m1, Sigma = S1)
m2 <- c(1.5, −1.5)  # lower right
S2 <- matrix(c(0.5, −0.08, −0.08, 0.2), ncol = 2)
clus2 <- mvrnorm(n = n, mu = m2, Sigma = S2)
m3 <- c(−1.5, 1.5)  # upper left
S3 <- matrix(c(0.1, 0.03, 0.03, 0.1), ncol = 2)
clus3 <- mvrnorm(n = n, mu = m3, Sigma = S3)
m4 <- c(−1.5, −1.5) # lower left
S4 <- matrix(c(0.8, 0.50, 0.50, 0.8), ncol = 2)
clus4 <- mvrnorm(n = n, mu = m4, Sigma = S4)
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