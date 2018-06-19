% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = binary_smc(objective, inputs)
% BINARY_SMC: Function explores the binary space of models and returns the
% expected value for each binary variable.

addpath(genpath('algorithms/SMC_Code'))

%% Declare Parameters

% Number of SMC particles
n_models     = 10;

% ESS reduction & final and initial tolerance
init_rho     = 1e-5;
alpha        = 0.90;
delta        = 0.2;

% find maximum number of objective evaluations
max_eval = inputs.evalBudget;

%% Setup SMC Simulation

ps = struct;
ps.n_vars = inputs.n_vars;

% Save initial eps
ps.rho = init_rho;

% Save alpha
ps.alpha_star = alpha;

% Initialize mh_accept
ps.mh_accept = 1;

%% Run first iteration

smc_iter = tic;

% Sample initial models
ps.models = sample_models(n_models, ps.n_vars);

% Compute model distances and model_prob
ps.objective = @(x) -1*objective(x);
ps.model_val = ps.objective(ps.models);

% Count function count
ps.total_fn_count = n_models;

% Set initial importance weights
ps.weights = 1/n_models*ones(n_models,1);

% Initialize logistic_A
ps.logistic_A = eye(ps.n_vars);

% Initialize counter
counter = 1;

% Setup cells to store models and weights
models_iter  = {ps.models};
weights_iter = {ps.weights};

% Set initial alpha and weights
ps = find_tolerance(ps);

% Setup vectors to store ESS and PDIV
ps.ess  = eff_sample_size(ps.weights);
ps.pdiv = unique_particles(ps.models);

% Setup variables for best current solution
[~, opt_model_idx] = max(ps.weights);
model_best = ps.models(opt_model_idx,:);
obj_best   = -1*ps.objective(model_best);

% Save model, model_obj, and runtime
model_iter = repmat(model_best, n_models, 1);
obj_iter   = repmat(obj_best, n_models, 1);
time_iter  = interp1([0,1],[0,toc(smc_iter)],linspace(1/n_models,1,n_models))';

%% Run Simulations

% Run algorithm until convergence of rho_k
while(ps.pdiv > delta)

    smc_iter = tic;
    
    % Fit logistic regression model
    ps = fit_logistic(ps);

	% Resample models
    ps = smc_resample(ps);

    % Move particles with the logistic regression model
    ps = move_particles(ps);
    
    % Find step length and compute weights
    ps = find_tolerance(ps);
    
    % Update counter
    counter = counter + 1;

    % Update ESS and PDIV
    ps.ess  = eff_sample_size(ps.weights);
    ps.pdiv = unique_particles(ps.models);

    % Add current models to cell
    models_iter{counter}  = ps.models;
    weights_iter{counter} = ps.weights;

    % Check best current solution
    [~, opt_model_idx] = max(ps.weights);
    model_temp = ps.models(opt_model_idx,:);
    obj_temp = -1*ps.objective(model_temp);

    % Update model_best & obj_best variables
    if obj_temp <= obj_best
        model_best = model_temp;
        obj_best   = obj_temp;
    end

    % Account for total number of function evaluations
    % when saving model_best, obj_best, and runtime 
    evals_iter = ps.total_fn_count(end) - ps.total_fn_count(end-1);
    model_iter_app = repmat(model_best, evals_iter, 1);
    obj_iter_app   = repmat(obj_best, evals_iter, 1);
    time_iter_app  = interp1([0,1],[0,toc(smc_iter)],linspace(1/evals_iter,1,evals_iter))';

    % Save 
    model_iter = [model_iter; model_iter_app];
    obj_iter   = [obj_iter; obj_iter_app];
    time_iter  = [time_iter; time_iter_app]; 

    % If budget of evaluations is reached, exit loop
    if ps.total_fn_count(end) > max_eval
	    break
    end

end

% save outputs
output = struct;
output.objVals  = obj_iter(1:max_eval); 
output.optModel = model_iter(1:max_eval,:);
output.runTime  = time_iter(1:max_eval);

% -- END OF FILE --
