% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = BOCS_categorical(model, penalty, inputs, order, acquisition_fn)
% BOCS_CATEGORICAL: Function runs categorical optimization using simulated 
% annealing on the model drawn from the distribution over beta parameters

% Set the number of SA reruns
SA_reruns  = 5;

% Extract inputs
nVars_     = inputs.n_vars;
estimator_ = inputs.estimator;

% Train initial statistical model
LR = LinReg(nVars_, order, estimator_);
LR = LR.train(inputs);

% Find number of iterations based on total budget
n_init = size(inputs.x_vals,1);
n_iter = inputs.evalBudget - n_init;

% Declare vector to store results
model_iter = zeros(n_iter, nVars_);
obj_iter   = zeros(n_iter,1);
time_iter  = zeros(n_iter,1);

alpha_all = zeros(n_iter, LR.nCoeffs+1);

for t=1:n_iter

    bocs_iter = tic;

    % Draw alpha vector
    alpha_t = LR.sampleCoeff();
    alpha_all(t,:) = alpha_t';

    % Run SA optimization
    if strcmp(acquisition_fn, 'SA')

        % Setup statistical model for SA
        stat_model = @(x) LR.surrogate_model(x, alpha_t) + penalty(x);

        SA_model = zeros(SA_reruns, nVars_);
        SA_obj   = zeros(SA_reruns, 1);

        for j=1:SA_reruns
            out = simulated_annealing_categorical(stat_model, inputs);
            SA_model(j,:) = out.optModel(end,:);
            SA_obj(j)     = out.objVals(end);
        end

        % Find optimal solution
        [~, min_idx] = min(SA_obj);
        x_new = SA_model(min_idx,:);

    else
        error('Acquisition function is not implemented for BOCS_categorical')
    end

    % evaluate model objective at new evaluation point
    y_new = model(x_new);

    % Update inputs struct
    inputs.x_vals = [inputs.x_vals; x_new];
    inputs.y_vals = [inputs.y_vals; y_new];
    inputs.init_cond = x_new;

    % re-train linear model
    LR = LR.train(inputs);

    % Save results for optimal model
    model_iter(t,:) = x_new;
    obj_iter(t)     = y_new + penalty(x_new);
    time_iter(t)    = toc(bocs_iter);

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

output.alpha_all = alpha_all;

end

function output = simulated_annealing_categorical(objective, inputs)
% SIMULATED_ANNEALING_CATEGORICAL: Function runs simulated annealing 
% algorithm for optimizing functions with categorical inputs. The function
% returns optimum models and min objective values found at each iteration
%
% Inputs: objective - function handle
%         inputs - struct containing n_vars, evalBudget and domains for inputs

% Extract inputs
n_vars  = inputs.n_vars;
n_iter  = inputs.evalBudget;
domains = inputs.domains;

% Declare vectors to save solutions
model_iter = zeros(n_iter,n_vars);
obj_iter   = zeros(n_iter,1);
time_iter  = zeros(n_iter,1);

% Set cooling schedule
cool = @(T) (.8*T);

% Set parameters
T = 1;
counter  = 0;
success  = 0;

% Set initial condition and evaluate objective
old_x   = sample_models(1, n_vars, domains);
old_obj = objective(old_x);

% Set best_x and best_obj
best_x   = old_x;
best_obj = old_obj;

% Run simulated annealing
while (counter < n_iter)

    % Increment counter
    counter = counter + 1;
    sa_iter = tic;
    
    % Decrease T according to cooling schedule
    T = cool(T);

    % Find new sample
    flip_bit = randi(n_vars);
    new_x = old_x;
    new_x(flip_bit) = domains{flip_bit}(randi(length(domains{flip_bit})));

    % Evaluate objective function
    new_obj = objective(new_x);
    
    % Update current solution iterate
    if (new_obj < old_obj) || (rand < exp( (old_obj - new_obj)/T ))
        old_x = new_x;
        old_obj = new_obj;
        success = success + 1;
    end

    % Update best solution
    if new_obj < best_obj
        best_x = new_x;
        best_obj = new_obj;
    end  

    % save solution
    model_iter(counter,:) = best_x;
    obj_iter(counter)     = best_obj;
    time_iter(counter)    = toc(sa_iter);

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end