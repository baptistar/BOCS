% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = SA_spears(objective, inputs)
% SIMULATED_ANNEALING_SPEARS: Function runs simulated annealing algorithm using the
% SPEARS algorithm for optimizing binary functions (MAX-SAT). The reference with
% the description of the algorithm and the parameters can be found in:
%
% An Experimental Evaluation of Fast Approximation Algorithms for the Maximum
% Satisfiability Problem by: MATTHIAS POLOCZEK and DAVID P. WILLIAMSON

% Extract n_vars
n_vars = inputs.n_vars;

% Set temperature limits
max_temp = 10;
min_temp = 0.01;

% Set counter and T
counter  = 0;
T = max_temp;

% Set initial condition and evaluate objective function
new_x   = sample_models(1,n_vars);
new_obj = objective(new_x);

% Set best variables
best_x   = new_x;
best_obj = new_obj;

% Declare vectors to save solutions
model_iter = zeros(0,n_vars);
obj_iter   = zeros(0,1);
time_iter  = zeros(0,1);

% Run simulated annealing
while(T >= min_temp)

    % Increment counter
    counter = counter + 1;
    sa_iter = tic;

    % Decrease T according to cooling schedule
    T = T*exp(-1/n_vars);

    %% Compute change from flipping each variable
    for i=1:n_vars

        % compute change in objective from flipping bit
        temp_x  = new_x; temp_x(i) = 1 - temp_x(i);
        temp_df = objective(temp_x) - new_obj;

        % Compute probability of flipping variable
        p = 1/(1 + exp(-temp_df/T));
        if rand < p
            new_x = temp_x;
        end

    end

    % Evaluate objective function at current solution
    new_obj = objective(new_x);

    % Update best solution
    if new_obj < best_obj
        best_x = new_x;
        best_obj = new_obj;
    end  

    % save solution
    model_iter = [model_iter; best_x];
    obj_iter   = [obj_iter; best_obj];
    time_iter  = [time_iter; toc(sa_iter)];

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end