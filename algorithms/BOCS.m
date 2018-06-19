% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = BOCS(model, penalty, inputs, order, acquisition_fn)
% BOCS: Function runs binary optimization using simulated annealing on
% the model drawn from the distribution over beta parameters

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

for t=1:n_iter
	
	bocs_iter = tic;

	% Draw alpha vector
	alpha_t = LR.sampleCoeff();

	% Run SA optimization
	if strcmp(acquisition_fn, 'SA') || strcmp(acquisition_fn, 'spears')

		% Setup statistical model for SA
		stat_model = @(x) LR.surrogate_model(x, alpha_t) + penalty(x);

		SA_model = zeros(SA_reruns, nVars_);
		SA_obj   = zeros(SA_reruns, 1);

		for j=1:SA_reruns
			if strcmp(acquisition_fn, 'SA')
				out = simulated_annealing(stat_model, inputs);
			elseif strcmp(acquisition_fn, 'spears')
				out = SA_spears(stat_model, inputs);
			end
			SA_model(j,:) = out.optModel(end,:);
			SA_obj(j)     = out.objVals(end);
		end

		% Find optimal solution
		[~, min_idx] = min(SA_obj);
		x_new = SA_model(min_idx,:);

	% Run semidefinite relaxation
	elseif strcmp(acquisition_fn, 'sdp')
		[x_new, ~] = sdp_relaxation(alpha_t, inputs);
	else
		error('Acquisition Function is not Implemented!')
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
	obj_iter(t)   	= y_new + penalty(x_new);
	time_iter(t)    = toc(bocs_iter);

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end