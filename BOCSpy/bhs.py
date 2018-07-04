# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np

def bhs(Xorg, yorg, nsamples, burnin, thin):
    # Implementation of the Bayesian horseshoe linear regression hierarchy.
    # Parameters:
    #   Xorg     = regressor matrix [n x p]
    #   yorg     = response vector  [n x 1]
    #   nsamples = number of samples for the Gibbs sampler (nsamples > 0)
    #   burnin   = number of burnin (burnin >= 0)
    #   thin     = thinning (thin >= 1)
    #
    # Returns:
    #   beta     = regression parameters  [p x nsamples]
    #   b0       = regression param. for constant [1 x nsamples]
    #   s2       = noise variance sigma^2 [1 x nsamples]
    #   t2       = hypervariance tau^2    [1 x nsamples]
    #   l2       = hypervariance lambda^2 [p x nsamples]
    #  
    #
    # Example:
    # % Load a dataset:
    # load hald;        
    # % Run horseshoe sampler. Normalising the data is not required.
    # [beta, b0] = bhs(ingredients, heat, 1000, 100, 10);    
    # % Plot the samples of the regression coefficients:
    # boxplot(beta', 'labels', {'tricalcium aluminate','tricalcium silicate',...
    #   'tetracalcium aluminoferrite', 'beta-dicalcium silicate'});
    # title('Bayesian linear regression with the horseshoe hierarchy');
    # xlabel('Predictors');
    # ylabel('Beta');
    # grid;
    #
    #
    # References:
    # A simple sampler for the horseshoe estimator
    # E. Makalic and D. F. Schmidt
    # arXiv:1508.03884, 2015
    #
    # The horseshoe estimator for sparse signals
    # C. M. Carvalho, N. G. Polson and J. G. Scott
    # Biometrika, Vol. 97, No. 2, pp. 465--480, 2010
    #
    # (c) Copyright Enes Makalic and Daniel F. Schmidt, 2015
    # Adapted to python by Ricardo Baptista, 2018

    n, p = Xorg.shape

    # Normalize data
    X, _, _, y, muY = standardise(Xorg, yorg)

    # Return values
    beta = np.zeros((p, nsamples))
    s2 = np.zeros((1, nsamples))
    t2 = np.zeros((1, nsamples))
    l2 = np.zeros((p, nsamples))

    # Initial values
    sigma2  = 1.
    lambda2 = np.random.uniform(size=p)
    tau2    = 1.
    nu      = np.ones(p)
    xi      = 1.

    # pre-compute X'*X (used with fastmvg_rue)
    XtX = np.matmul(X.T,X)  

    # Gibbs sampler
    k = 0
    iter = 0
    while(k < nsamples):

        # Sample from the conditional posterior distribution
        sigma = np.sqrt(sigma2)
        Lambda_star = tau2 * np.diag(lambda2)
        # Determine best sampler for conditional posterior of beta's
        if (p > n) and (p > 200):
            b = fastmvg(X/sigma, y/sigma, sigma2*Lambda_star)
        else:
            b = fastmvg_rue(X/sigma, XtX/sigma2, y/sigma, sigma2*Lambda_star)

        # Sample sigma2
        e = y - np.dot(X,b)
        shape = (n + p) / 2.
        scale = np.dot(e.T,e)/2. + np.sum(b**2/lambda2)/tau2/2.
        sigma2 = 1. / np.random.gamma(shape, 1./scale)

        # Sample lambda2
        scale = 1./nu + b**2./2./tau2/sigma2
        lambda2 = 1. / np.random.exponential(1./scale)

        # Sample tau2
        shape = (p + 1.)/2.
        scale = 1./xi + np.sum(b**2./lambda2)/2./sigma2
        tau2 = 1. / np.random.gamma(shape, 1./scale)

        # Sample nu
        scale = 1. + 1./lambda2
        nu = 1. / np.random.exponential(1./scale)

        # Sample xi
        scale = 1. + 1./tau2
        xi = 1. / np.random.exponential(1./scale)

        # Store samples
        iter = iter + 1;
        if iter > burnin:
            # thinning
            if (iter % thin) == 0:
                beta[:,k] = b
                s2[:,k]   = sigma2
                t2[:,k]   = tau2
                l2[:,k]   = lambda2
                k         = k + 1

    # Re-scale coefficients
    #div_vector = np.vectorize(np.divide)
    #beta = div_vector(beta.T, normX)
    #b0 = muY-np.dot(muX,beta)
    b0 = muY

    return (beta, b0, s2, t2, l2)

def fastmvg(Phi, alpha, D):
    # Fast sampler for multivariate Gaussian distributions (large p, p > n) of 
    #  the form N(mu, S), where
    #       mu = S Phi' y
    #       S  = inv(Phi'Phi + inv(D))
    # Reference: 
    #   Fast sampling with Gaussian scale-mixture priors in high-dimensional 
    #   regression, A. Bhattacharya, A. Chakraborty and B. K. Mallick
    #   arXiv:1506.04778

    n, p = Phi.shape

    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi,u) + delta
    #w = np.linalg.solve(np.matmul(np.matmul(Phi,D),Phi.T) + np.eye(n), alpha - v)
    #x = u + np.dot(D,np.dot(Phi.T,w))
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:,np.newaxis])
    w = np.linalg.solve(np.matmul(Phi,Dpt) + np.eye(n), alpha - v)
    x = u + np.dot(Dpt,w)

    return x

def fastmvg_rue(Phi, PtP, alpha, D):
    # Another sampler for multivariate Gaussians (small p) of the form
    #  N(mu, S), where
    #  mu = S Phi' y
    #  S  = inv(Phi'Phi + inv(D))
    #
    # Here, PtP = Phi'*Phi (X'X is precomputed)
    #
    # Reference:
    #   Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal
    #   of the Royal Statistical Society: Series B (Statistical Methodology) 
    #   63, 325-338

    p = Phi.shape[1]
    Dinv = np.diag(1./np.diag(D))

    # regularize PtP + Dinv matrix for small negative eigenvalues
    try:
        L = np.linalg.cholesky(PtP + Dinv)
    except:
        mat  = PtP + Dinv
        Smat = (mat + mat.T)/2.
        maxEig_Smat = np.max(np.linalg.eigvals(Smat))
        L = np.linalg.cholesky(Smat + maxEig_Smat*1e-15*np.eye(Smat.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T,alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))

    x = m + w

    return x

def standardise(X, y):
    # Standardize the covariates to have zero mean and x_i'x_i = 1

    # set params
    n = X.shape[0]
    meanX = np.mean(X, axis=0)
    stdX  = np.std(X, axis=0) * np.sqrt(n)

    # Standardize X's
    #sub_vector = np.vectorize(np.subtract)
    #X = sub_vector(X, meanX)
    #div_vector = np.vectorize(np.divide)
    #X = div_vector(X, stdX)

    # Standardize y's
    meany = np.mean(y)
    y = y - meany

    return (X, meanX, stdX, y, meany)

# -- END OF FILE --