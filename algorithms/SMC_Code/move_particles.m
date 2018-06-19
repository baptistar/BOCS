% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function particle_system = move_particles(particle_system)
% MOVE_PARTICLES: Function moves the current particles according to an MCMC
% kernel with an adaptive logistic regression proposal.

% Extract function inputs
n_vars     = particle_system.n_vars;
model_orig = particle_system.models;
model_val  = particle_system.model_val;
rho        = particle_system.rho;
logistic_A = particle_system.logistic_A;

% Determine the number of current models, and dimensions
[n_models, ~] = size(model_orig);

% Compute the old particle diversity ratio
diversity_old = unique_particles(model_orig);

% Declare vector for accept_ratios
mh_accept_vect = [];

total_fn_count = particle_system.total_fn_count(end);

iter_temp = 0;
while(1)
    
    % Generate matrix to store new samples and data
    new_model = zeros(n_models, n_vars);
    new_mval  = zeros(n_models, 1);

    % Declare vector to store acceptance ratio
    accept_ratio = 0;
    
    for i=1:n_models
            
        % Propose new model
        [prop_model, ~] = logistic_sample(n_vars, logistic_A);
        
        % Evaluate proposal density under both models
        q_new = evaluate_logit(particle_system, prop_model);
        q_old = evaluate_logit(particle_system, model_orig(i,:));

        % Evaluate probability function for both models
        total_fn_count = total_fn_count + 1;
        model_obj_new = particle_system.objective(prop_model);
        log_post_new = rho*model_obj_new;
        log_post_old = rho*model_val(i);

        % Compute acceptance probability
        accept_prob = exp(log_post_new - log_post_old)*(q_old/q_new);

        % Accept proposed sample/data or assign old values
        if rand < min(1,accept_prob)
            new_model(i,:) = prop_model;
            new_mval(i)    = model_obj_new;
            accept_ratio   = accept_ratio + 1;
        else
            new_model(i,:) = model_orig(i,:);
            new_mval(i)    = model_val(i);
        end
    
    end

    %% Update Sample Information

    % Update variables
    model_orig     = new_model;
    model_val      = new_mval;
    mh_accept_vect = [mh_accept_vect, accept_ratio/n_models];

    % Compute the new particle diversity
    diversity_new = unique_particles(new_model);

    % Compare the diversity
    if (abs(diversity_new - diversity_old) < 0.02 || diversity_new > 0.95)
        break
    end

    % Update diversity_old
    diversity_old = diversity_new;

    iter_temp = iter_temp + 1;
    if iter_temp > 100
        break
    end

end

% Update particle_system
particle_system.models     = model_orig;
particle_system.model_val  = model_val;
particle_system.mh_accept  = min(mh_accept_vect);
particle_system.total_fn_count = [particle_system.total_fn_count, total_fn_count];

end
