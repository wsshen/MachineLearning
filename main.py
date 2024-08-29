import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %matplotlib inline
plt.style.use('ggplot')
np.random.seed(1234)

np.set_printoptions(formatter={'all':lambda x: '%.3f' %x})

# from IPython.display import Image
# from numpy.core.umath_tests import matrix_multiply as mm

from scipy.optimize import minimize
from scipy.stats import bernoulli,binom

m = 10
theta_A = 0.8
theta_B = 0.3
theta_0 = [theta_A,theta_B]

coin_A = bernoulli(theta_A)
coin_B = bernoulli(theta_B)

xs = map(sum,[coin_A.rvs(m),coin_A.rvs(m),coin_B.rvs(m),coin_A.rvs(m),coin_B.rvs(m)])
zs = [0,0,1,0,1]

xs = np.array(xs)
bnds = [(0,1),(0,1)]
def neg_loglike(m,xs,):
    return np.sum(np.log(xs))
minimize(fun,[0.5,0.5],args=(m,xs,zs),bounds=bnds,method='tnc',options={'maxiter':100})

