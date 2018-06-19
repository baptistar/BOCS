%% Implementation of the Bayesian horseshoe linear regression hierarchy.
%
% function [beta, b0, s2, t2, l2] = bhs(Xorg, yorg, nsamples, burnin, thin)
%
% Parameters:
%   Xorg     = regressor matrix [n x p]
%   yorg     = response vector  [n x 1]
%   nsamples = number of samples for the Gibbs sampler (nsamples > 0)
%   burnin   = number of burnin (burnin >= 0)
%   thin     = thinning (thin >= 1)
%
% Returns:
%   beta     = regression parameters  [p x nsamples]
%   b0       = regression param. for constant [1 x nsamples]
%   s2       = noise variance sigma^2 [1 x nsamples]
%   t2       = hypervariance tau^2    [1 x nsamples]
%   l2       = hypervariance lambda^2 [p x nsamples]
%  
%
% Example:
% % Load a dataset:
% load hald;        
% % Run horseshoe sampler. Normalising the data is not required.
% [beta, b0] = bhs(ingredients, heat, 1000, 100, 10);    
% % Plot the samples of the regression coefficients:
% boxplot(beta', 'labels', {'tricalcium aluminate', 'tricalcium silicate', 'tetracalcium aluminoferrite', 'beta-dicalcium silicate'});
% title('Bayesian linear regression with the horseshoe hierarchy');
% xlabel('Predictors');
% ylabel('Beta');
% grid;
%
%
% References:
% A simple sampler for the horseshoe estimator
% E. Makalic and D. F. Schmidt
% arXiv:1508.03884, 2015
%
% The horseshoe estimator for sparse signals
% C. M. Carvalho, N. G. Polson and J. G. Scott
% Biometrika, Vol. 97, No. 2, pp. 465--480, 2010
%
% (c) Copyright Enes Makalic and Daniel F. Schmidt, 2015
function [beta, b0, s2, t2, l2] = bhs(Xorg, yorg, nsamples, burnin, thin)

[n, p] = size(Xorg);

%% Normalise data
[X, ~, ~, y, muY]=standardise(Xorg, yorg);

%% Return values
beta = zeros(p, nsamples);
s2 = zeros(1, nsamples);
t2 = zeros(1, nsamples);
l2 = zeros(p, nsamples);

%% Initial values
sigma2 = 1;
lambda2 = rand(p, 1);
tau2 = 1;
nu = ones(p,1);
xi = 1;

%% Determine best sampler for conditional posterior of beta's
if((p > n) && (p > 200))
    XtX = [];               % may consider precomputing (x_i x_i^t) 
    betasample = @fastmvg;  % good when p > n and p is large
else
    XtX = X'*X;             % pre-compute X'*X
    betasample = @fastmvg_rue;
end

%% Gibbs sampler
k = 0;
iter = 0;
while(k < nsamples)

    %% Sample from the conditional posterior dist. for beta
    sigma = sqrt(sigma2);
    Lambda_star = tau2 * diag(lambda2);
    b = betasample(X ./ sigma, XtX ./ sigma2, y ./ sigma, sigma2 * Lambda_star);

    %% Sample sigma2
    e = y - X*b;
    shape = (n + p) / 2;
    scale = e'*e/2 + sum(b.^2 ./ lambda2)/tau2/2;
    sigma2 = 1 / gamrnd(shape, 1/scale);
    
    %% Sample lambda2
    scale = 1./nu + b.^2./2./tau2./sigma2;
    lambda2 = 1 ./ exprnd(1./scale);
    
    %% Sample tau2
    shape = (p + 1)/2;
    scale = 1/xi + sum(b.^2./lambda2)/2/sigma2;
    tau2 = 1 / gamrnd(shape, 1/scale);
    
    %% Sample nu
    scale = 1 + 1./lambda2;
    nu = 1 ./ exprnd(1./scale);
    
    %% Sample xi
    scale = 1 + 1/tau2;
    xi = 1 / exprnd(1/scale);
  
    %% Store samples    
    iter = iter + 1;
    if(iter > burnin)
        % thinning
        if(mod(iter,thin) == 0)
            k = k + 1;
            beta(:,k) = b;
            s2(k) = sigma2;
            t2(k) = tau2;
            l2(:,k) = lambda2;
        end
    end
end


%% Re-scale coefficients
%beta = bsxfun(@rdivide, beta, normX');
%b0 = muY-muX*beta;
b0 = muY;

end

%% Fast sampler for multivariate Gaussian distributions (large p, p > n) of the form
%  N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
% Reference: 
%   Fast sampling with Gaussian scale-mixture priors in high-dimensional regression
%   A. Bhattacharya, A. Chakraborty and B. K. Mallick
%   arXiv:1506.04778
function x = fastmvg(Phi, ~, alpha, D)

[n,p] = size(Phi);

d = diag(D);
u = randn(p,1) .* sqrt(d);
delta = randn(n,1);
v = Phi*u + delta;
%w = (Phi*D*Phi' + eye(n)) \ (alpha - v);
%x = u + D*Phi'*w;
Dpt = bsxfun(@times, Phi', d);
w = (Phi*Dpt + eye(n)) \ (alpha - v);
x = u + Dpt*w;

end

%% Another sampler for multivariate Gaussians (small p) of the form
%  N(mu, S), where
%       mu = S Phi' y
%       S  = inv(Phi'Phi + inv(D))
%
% Here, PtP = Phi'*Phi (X'X is precomputed)
%
% Reference:
%   Rue, H. (2001). Fast sampling of gaussian markov random fields. Journal of the Royal
%   Statistical Society: Series B (Statistical Methodology) 63, 325–338.
function x = fastmvg_rue(Phi, PtP, alpha, D)

p = size(Phi,2);

Dinv = diag(1./diag(D));
%L = chol(Phi'*Phi + Dinv, 'lower');
% regularize PtP + Dinv matrix for small negative eigenvalues
try
    L = chol(PtP + Dinv, 'lower');
catch
    mat = PtP + Dinv;
    Smat = (mat + mat')/2;
    L = chol(Smat + max(eig(Smat))*(1e-15)*eye(size(Smat,1)),'lower');
end
v = L \ (Phi'*alpha);
m = L' \ v;
w = L' \ randn(p,1);

x = m + w;

end

%% Standardise the covariates to have zero mean and x_i'x_i = 1
function [X,meanX,stdX,y,meany] = standardise(X, y)

%% params
n = size(X, 1);
meanX = mean(X);
stdX = std(X,1) * sqrt(n);

%% Standardise Xs
%X = bsxfun(@minus,X,meanX);
%X = bsxfun(@rdivide,X,stdX);

%% Standardise ys (if neccessary)
if(nargin == 2)
    meany = mean(y);
    y = y - meany;
end;

%% done
end
