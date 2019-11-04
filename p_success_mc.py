from scipy.stats import norm
import numpy as np
import random

def p_success_mc(n, m, alpha, xi, Nsim = 10000):

    #This function returns the ordinal optimisation probability of success for
    #the Gaussian case, using Monte-Carlo simulation

    count = 0

    X_thres = norm.ppf(alpha)

    for j in range(Nsim):
        X = norm.rvs(size=n)
        Y = xi*norm.rvs(size=n)
        Z = X + Y
    
        #Count number of successes
        ind = np.argsort(Z)
        if (min(X[ind[0:m]]) <= X_thres):
            count += 1
    
    p = count/Nsim
    #Return the standard error as a second argument
    se = np.sqrt(p*(1 - p)/Nsim)
    
    return p, se
