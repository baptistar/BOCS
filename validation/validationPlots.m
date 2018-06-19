% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function validationPlots(inputs_all, n_train, n_test, n_proc)
% Script generates validation plots that compare the effect of 
% increasing set size and order of the regression model.
% The script generates the data by running compare_models

% Extract n_func, n_vars, order
n_func = length(inputs_all);
n_vars = inputs_all{1}.n_vars;
ordstr = num2str(inputs_all{1}.order);

% Save total errors
bayes_err = zeros(length(n_train), n_test, n_func);
mle_err   = zeros(length(n_train), n_test, n_func);
hs_err    = zeros(length(n_train), n_test, n_func);

% Generate file_head figure name
file_head = ['../validation/figures/' inputs_all{1}.name];

% plot results for each test function
c = parcluster('local');
c.NumWorkers = n_proc;
parpool(n_proc);

parfor t=1:n_func

	fprintf(['Model ' inputs_all{1}.name, ' - Order: ' ordstr ...
		', Test: ' num2str(t) '/' num2str(n_func) '\n'])

	% Extract inputs_t
	inputs_t = inputs_all{t};

    % Generate test samples
    inputs_t.x_test = sample_models(n_test, n_vars);
    inputs_t.y_test = inputs_t.model(inputs_t.x_test);

	% Generate all training samples 
	x_train = sample_models(max(n_train), n_vars);
	y_train = inputs_t.model(x_train);

    % Generate temporary arrays to save data
    bayes_t = zeros(length(n_train), n_test);
    mle_t   = zeros(length(n_train), n_test);
    hs_t    = zeros(length(n_train), n_test);
    
	for s=1:length(n_train)

		% Save training samples
		inputs_t.x_vals = x_train(1:n_train(s),:);
		inputs_t.y_vals = y_train(1:n_train(s));

		% compute and save results
		[bayes_t(s,:), mle_t(s,:), hs_t(s,:)] = compare_models(inputs_t);
        
    end
    
    % Save results
    bayes_err(:,:,t) = bayes_t;
    mle_err(:,:,t)   = mle_t;
    hs_err(:,:,t)    = hs_t;

end

delete(gcp('nocreate'))

% save results in file
res_file = ['../validation/results/' inputs_all{1}.name '_order' ordstr];
save(res_file)

end