import numpy as np
from scipy.stats import norm
from scipy.stats import binom
from scipy.integrate import quad
from scipy.special import loggamma

def lefttrunc_normpdf(x, thres):
    #This function computes the densities of the left-truncated
    #standard Gaussian with the supplied threshold at the given points
        
    densities = (x >= thres)*norm.pdf(x)/(1 - norm.cdf(thres))
    
    return densities

def righttrunc_normpdf(x, thres):
    #This function computes the densities of the right-truncated
    #standard Gaussian with the supplied threshold at the given points
        
    densities = (x <= thres)*norm.pdf(x)/norm.cdf(thres)
    
    return densities
    
def sumrvs_pdf_lower_integrand(x, z, alpha, xi):
    
    return norm.pdf(z - x, loc=0, scale=xi)*righttrunc_normpdf(x, norm.ppf(alpha))
   
def sumrvs_pdf_lower(x, alpha, xi):
    #This function computes the densities of the sum distribution of a
    #scaled Gaussian (with std. dev. xi) with the right-truncated
    #standard Gaussian at the alpha*100 percentile
        
    #deal with x possibly being scalar
    x = np.asarray(x)
    x = x * np.ones(x.size)
    densities = np.zeros(x.size)
    
    for i in range(x.size):
        densities[i] = quad(sumrvs_pdf_lower_integrand, -np.inf, norm.ppf(alpha), args=(x[i], alpha, xi))[0]
        
    return densities

def sumrvs_cdf_lower_integrand(x, z, alpha, xi):
    
    return norm.cdf(z - x, loc=0, scale=xi)*righttrunc_normpdf(x, norm.ppf(alpha))
   
def sumrvs_cdf_lower(x, alpha, xi):
    #This function computes the cumulative probabilities of the sum distribution of a
    #scaled Gaussian (with std. dev. xi) with the right-truncated
    #standard Gaussian at the alpha*100 percentile
        
    #deal with x possibly being scalar
    x = np.asarray(x)
    x = x * np.ones(x.size)
    densities = np.zeros(x.size)
    
    for i in range(x.size):
        densities[i] = quad(sumrvs_cdf_lower_integrand, -np.inf, norm.ppf(alpha), args=(x[i], alpha, xi))[0]
        
    return densities

def sumrvs_pdf_upper_integrand(x, z, alpha, xi):
    
    return norm.pdf(z - x, loc=0, scale=xi)*lefttrunc_normpdf(x, norm.ppf(alpha))
   
def sumrvs_pdf_upper(x, alpha, xi):
    #This function computes the densities of the sum distribution of a
    #scaled Gaussian (with std. dev. xi) with the left-truncated
    #standard Gaussian at the alpha*100 percentile
    
    #deal with x possibly being scalar
    x = np.asarray(x)
    x = x * np.ones(x.size)
    densities = np.zeros(x.size)
    
    for i in range(x.size):
        densities[i] = quad(sumrvs_pdf_upper_integrand, norm.ppf(alpha), np.inf, args=(x[i], alpha, xi))[0]
        
    return densities

def sumrvs_cdf_upper_integrand(x, z, alpha, xi):
    
    return norm.cdf(z - x, loc=0, scale=xi)*lefttrunc_normpdf(x, norm.ppf(alpha))
   
def sumrvs_cdf_upper(x, alpha, xi):
    
    #This function computes the cumulative probabilities of the sum distribution of a
    #scaled Gaussian (with std. dev. xi) with the left-truncated
    #standard Gaussian at the alpha*100 percentile
        
    #deal with x possibly being scalar
    x = np.asarray(x)
    x = x * np.ones(x.size)
    densities = np.zeros(x.size)

    for i in range(x.size):
        densities[i] = quad(sumrvs_cdf_upper_integrand, norm.ppf(alpha), np.inf, args=(x[i], alpha, xi))[0]
        
    return densities

def Zpdf_lower(x, n, m, g, alpha, xi):
    #This function returns the density of Z_{1} at the supplied points

    F = sumrvs_cdf_lower(x, alpha, xi)
    f = sumrvs_pdf_lower(x, alpha, xi)
    
    densities = g*(1 - F)**(g - 1)*f;
    
    #O(1) computation but may run into numerical precision issues
    #densities = np.exp(np.log(g) + (g - 1)*np.log(1 - F) + np.log(f));
    
    return densities

def Zpdf_upper(x, n, m, g, alpha, xi):
    #This function returns the density of Z_{g + m} at the supplied points
    
    F = sumrvs_cdf_upper(x, alpha, xi)
    f = sumrvs_pdf_upper(x, alpha, xi)
    
    densities = np.exp(np.log(m) + loggamma(n - g + 1) - loggamma(m + 1) - loggamma(n - g - m + 1))*F**(m-1)*(1 - F)**(n-g-m)*f
    
    #O(1) computation but may run into numerical precision issues
    #densities = np.exp(np.log(m) + loggamma(n - g + 1) - loggamma(m + 1) - loggamma(n - g - m + 1)
    #                  + (m - 1)*np.log(F) + (n - g - m)*np.log(1 - F) + np.log(f))
    
    return densities

def Zcdf_upper(x, n, m, g, alpha, xi):
    #This function returns the cumulative probability of Z_{g + m} at the supplied points
    
    #deal with x possibly being scalar
    x = np.asarray(x)
    x = x * np.ones(x.size)
    cprobs = np.zeros(x.size)
    
    for i in range(x.size):
        cprobs[i] = quad(Zpdf_upper, -np.inf, x[i], args=(n, m, g, alpha, xi))[0]
    
    return cprobs

def p_success_numerical_integrand(x, n, m, g, alpha, xi):
    
    return np.multiply(Zcdf_upper(x, n, m, g, alpha, xi), Zpdf_lower(x, n, m, g, alpha, xi))

def p_success_numerical(n, m, alpha, xi):
    #This function returns the ordinal optimisation probability of success for
    #the Gaussian case, using numerical integration
    
    p = 1 - (1 - alpha)**n
    
    for g in range(1, n - m + 1):
        
        p -= binom.pmf(g, n, alpha)*quad(p_success_numerical_integrand, -np.inf, np.inf, args=(n, m, g, alpha, xi))[0]
    
    return p
