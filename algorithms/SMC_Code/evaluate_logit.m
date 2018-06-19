% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function q_theta = evaluate_logit(particle_system, model)
% EVALUATE_LOGIT: Function evaluates logistic regression model at the
% specific model and returns the value in q_theta

% Extract function inputs
logistic_A = particle_system.logistic_A;
models     = particle_system.models;

% Compute n_vars
n_vars = size(models,2);

% Define q_theta
q_theta = 1;

for i=1:n_vars
    
    % Compute probability for each component
    logit_p = logistic_A(i,i) + logistic_A(i,1:i-1)*model(1:i-1)';
    logit_p = 1/(1 + exp(-1*logit_p));
    
    % Multiply component probability to q_theta
    q_theta = q_theta*(logit_p^(model(i))*(1 - logit_p)^(1 - model(i)));

end