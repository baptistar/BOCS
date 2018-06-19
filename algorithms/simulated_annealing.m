% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = simulated_annealing(objective, inputs)
% SIMULATED_ANNEALING: Function runs simulated annealing algorithm for 
% optimizing binary functions. The function returns optimum models and min 
% objective values found at each iteration

% Extract inputs
n_vars = inputs.n_vars;
n_iter = inputs.evalBudget;

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
old_x   = sample_models(1,n_vars);
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
    new_x(flip_bit) = 1 - new_x(flip_bit);

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