% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function particle_system = smc_resample(particle_system)
% SMC_RESAMPLE: Function resamples binary particles (or models) based on
% the effective sample size at the current iteration.

% Extract inputs from particle_system
models     = particle_system.models;
model_val  = particle_system.model_val;
weights    = particle_system.weights;

% Generate new particles and data
[new_models, new_idx] = binary_resampling(models, weights);
n_models = size(new_models,1);

% Assign uniform weights
new_weights  = 1/n_models*ones(n_models,1);

% Re-order model data
model_val  = model_val(new_idx);

% Save models and weights
particle_system.models     = new_models;
particle_system.model_val  = model_val;
particle_system.weights    = new_weights;

end
