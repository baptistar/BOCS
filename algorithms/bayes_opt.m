% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = bayes_opt(objective, inputs)
% BAYES_OPT: Function runs MATLAB's bayesian optimization and
% returns optimum models and min objective values at each iteration

% Extract inputs
n_vars = inputs.n_vars;
n_iter = inputs.evalBudget;

% Setup x variables
x_vect = [];
for i=1:n_vars
    x_vect = [x_vect, optimizableVariable(['x' num2str(i)],[0,1],'Type','integer')];
end

% Run Bayes Opt
bo_time = tic;
results = bayesopt(@(x) bayes_obj(objective, x), x_vect, 'Verbose', 0, ...
	'AcquisitionFunctionName', 'expected-improvement', 'PlotFcn', [],...
	'MaxObjectiveEvaluations', n_iter);

% Extract trace of objective and models
obj_iter   = results.ObjectiveMinimumTrace;
model_iter = table2array(results.XTrace);

% Compute actual traces (using true values)
obj_unique = flipud(unique(obj_iter));
if length(obj_unique) > 1
	for k=1:length(obj_unique)-1
		idx1 = find(obj_iter == obj_unique(k),1);
		idx2 = find(obj_iter == obj_unique(k+1),1);
		model_iter(idx1:idx2-1,:) = repmat(model_iter(idx1,:),idx2-idx1,1);
	end
	num_final_mod = size(model_iter,1) - idx2 + 1;
	model_iter(idx2:end,:) = repmat(model_iter(idx2,:),num_final_mod,1);
elseif length(obj_unique) == 1
	idx = find(obj_iter == obj_unique,1);
	model_iter(1:n_iter,:) = repmat(model_iter(idx,:),n_iter,1);
end

% Save total runtime
total_time = toc(bo_time);

% Save time_iter
time_iter = repmat(total_time/n_iter,n_iter,1);

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end