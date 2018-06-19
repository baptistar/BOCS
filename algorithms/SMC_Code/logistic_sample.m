% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [binary_sample, p] = logistic_sample(n_vars, logistic_A)
% LOGISTIC_SAMPLE: Function computes binary sample from the implemented
% logistic regression method, as well as the probability of the sample
% (likelihood under the binary logistic regression proposal).

% Declare new binary sample
binary_sample = zeros(1,n_vars); 

% Declare new constant p
p = 1;

% Generate sample
for j=1:n_vars
    
    % Evaluate logistic regression function
    if j==1
        logistic_regressor = logistic_A(j,j);
    else
        logistic_regressor = logistic_A(j,j) + ...
                             logistic_A(j,1:j-1)*binary_sample(1:j-1)';
    end
    logistic_function = 1/(1 + exp(-logistic_regressor));

    % Generate sample from a bernoulli distribution
    binary_sample(j) = binornd(1,logistic_function);
    
    % Compute next success probability
    if (binary_sample(j) == 1)
        p = p*logistic_function;
    elseif (binary_sample(j) == 0)
        p = p*(1-logistic_function);
    end

end

end