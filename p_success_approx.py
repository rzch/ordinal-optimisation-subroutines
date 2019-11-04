import numpy as np
import random
import math
from scipy.stats import norm
from scipy.stats import mvn

def joint_asymp_order_stat(n, m, xi):
    #return the mean and covariance of the Gaussian approximation of the joint order statistics
    
    p = np.divide(np.subtract(n, np.arange(m, 0, -1)), n)
    
    sd = np.sqrt(1 + xi**2)
    
    means = norm.ppf(p, 0, sd)
    
    covmat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if (i <= j):
                covmat[i, j] = p[i]*(1 - p[j])/n/norm.pdf(means[i], 0, sd)/norm.pdf(means[j], 0, sd)
                covmat[j, i] = covmat[i, j]
    
    return means, covmat
    
def p_success_approx(n, m, alpha, xi):

    #This function returns the ordinal optimisation probability of success for
    #the Gaussian case, using the approximation formula

    a, A = joint_asymp_order_stat(n, m, xi)
    B = np.multiply(xi**2/(1 + xi**2), np.eye(m))
    C = np.multiply(1/(1 + xi**2), np.eye(m))

    #marginalise Gaussians
    means = np.multiply(1/(1 + xi**2), a)
    covmat = np.add(B, np.multiply(1/(1 + xi**2)**2, A))
    
    #lower and upper terminals of multivariate Gaussian CDF integral
    lower = np.full((m, 1), -np.inf)
    upper = np.full((m, 1), norm.ppf(1 - alpha))

    return 1 - mvn.mvnun(lower, upper, means, covmat)[0]
