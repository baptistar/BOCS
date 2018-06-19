% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [s_mean, s_cov, s_corr] = samp_stats(models, weights)
% SAMP_STATS: Function computes statistics of the current samples and
% weights for improving the efficiency of the binary logistic regression

% Determine the number of samples and dimensions
[~, n_vars] = size(models);

% Declare vector and matrices to store results
s_mean = zeros(n_vars,1);
s_cov  = zeros(n_vars,n_vars);
s_corr = zeros(n_vars,n_vars);

% Compute the sample mean
for i=1:n_vars
    s_mean(i) = sum(weights.*models(:,i));
end
    
% Compute the sample cov and corr
for i=1:n_vars
    for j=1:i
        s_cov(i,j)  = sum(weights.*models(:,i).*models(:,j));
        s_corr(i,j) = (s_cov(i,j) - s_mean(i)*s_mean(j))/...
                      sqrt(s_mean(i)*(1 - s_mean(i))*s_mean(j)*(1 - s_mean(j)));
    end
end