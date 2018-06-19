% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = local_search(objective, inputs)
% LOCAL_SEARCH: Function runs binary optimization by searching over the neighborhood
% of single model flips at each iteration

% Extract inputs
nVars = inputs.n_vars;
nEval = inputs.evalBudget;

% Generate initial condition and evaluate objective
model 	  = sample_models(1,nVars);
model_val = objective(model);

% determine the total number of iterations
nIter = ceil(nEval/nVars);

% Setup cells to store model, objective, and runtime
model_iter = zeros(nIter, nVars);
obj_iter   = zeros(nIter,1);
time_iter  = zeros(nIter,1);

% Declare counter
counter = 0;

for i=1:nIter

	% Update counter
	counter = counter + 1;

	ls_iter = tic;

	% Setup vector to store new objective values and difference
	new_obj  = zeros(nVars,1);
	diff_obj = zeros(nVars,1);

	for j=1:nVars
		
		% Setup new_model with one flipped variable
		new_model = model;
		new_model(j) = 1-new_model(j);

		% Evaluate objective
		new_obj(j)  = objective(new_model);
		diff_obj(j) = model_val - new_obj(j);

	end

	% Check if diff_obj is positive - improvement can be made
	if any(diff_obj > 0)

		% Choose optimal index to flip
    	[~, opt_idx] = max(diff_obj);
    	model(opt_idx) = 1 - model(opt_idx);
    	model_val = new_obj(opt_idx);

    end

	% Save models, model_obj, and runtime
	model_iter(counter,:) = model;
	obj_iter(counter)  = model_val;
	time_iter(counter) = toc(ls_iter);
	
end

% extend results
model_iter_new = zeros(nVars*nIter, nVars);
time_iter_new  = zeros(nVars*nIter, 1);
for i=1:nIter
	idx = nVars*(i-1)+1:nVars*i;
	model_iter_new(idx,:) = repmat(model_iter(i,:), nVars, 1);
	time_iter_new(idx) = interp1([0,1],[0,time_iter(i)],linspace(1/nVars,1,nVars));
end
obj_iter   = reshape(repmat(obj_iter, 1, nVars)', nVars*nIter, 1);

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter_new;
output.runTime  = time_iter_new;

end