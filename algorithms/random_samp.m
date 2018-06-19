% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = random_samp(objective, inputs)
% RANDOM_SAMP: Function generates random samples to find the minimum of an 
% objective function starting from the specified initial condition

% Extract n_vars
n_vars = inputs.n_vars;
n_iter = inputs.evalBudget;

% Generate initial condition and evaluate objective
model = sample_models(1,n_vars);
model_val = objective(model);

% Setup cells to store model, objective, and runtime
model_iter = zeros(n_iter, n_vars);
obj_iter   = zeros(n_iter,1);
time_iter  = zeros(n_iter,1);

% Declare counter
counter = 0;

for i=1:n_iter

	% Update counter
	counter = counter + 1;
	rand_iter = tic;

	% Sample random model and evaluate objective
	new_model = sample_models(1, n_vars);
	new_model_val = objective(new_model);

	% If model is better, update model, model_val 
	if new_model_val < model_val
		model = new_model;
		model_val = new_model_val;
	end

	% Save models, model_obj, and runtime
	model_iter(counter,:) = model;
	obj_iter(counter)  = model_val;
	time_iter(counter) = toc(rand_iter);

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end