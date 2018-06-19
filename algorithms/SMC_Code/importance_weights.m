% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function weights_new = importance_weights(particle_system, new_rho)
% IMPORTANCE_WEIGHTS: Function updates the importance weights for each
% binary model based on the geometric bridge model and the empirical
% distribution

% Extract inputs to function
models     = particle_system.models;
model_val  = particle_system.model_val;
old_wts    = particle_system.weights;
old_rho    = particle_system.rho;

% Determine the number of models and trials
[n_models, ~] = size(models);

% Declare a vector to store weights
weights_new = zeros(n_models,1);

for i=1:n_models

    % Evaluate posterior ratios: pi(M) \propto exp(rho*f(M))
    post_new_rho = new_rho*model_val(i);
    post_old_rho = old_rho*model_val(i);

    % Find the weights corresponding to each model
    weights_new(i) = old_wts(i)*exp(post_new_rho - post_old_rho);

end

% Check for NaNs and set weight to zero
nan_weight = isnan(weights_new);
weights_new(nan_weight) = 0;

% Renormalize weights
weights_new = weights_new/sum(weights_new);

end