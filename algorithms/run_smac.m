% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function output = run_smac(objective, inputs)
% SMAC: Function interfaces with pySMAC solver to find optimal solutions 
% using SMAC genetic algorithm

% define SMAC parameters
seed 		   = 1;
cutoff_time    = 86400;
rf_num_trees   = 100;
rf_bootstrap   = 1;
int_percentage = 0.0;

% extract n_vars
nVars = inputs.n_vars;
nIter = inputs.evalBudget;

% setup input parameters for SMAC
x0   = py.list({}); % real variables
xmin = py.list({}); % real variables
xmax = py.list({}); % real variables

x0_int   = zeros(1,nVars);
xmin_int = zeros(1,nVars);
xmax_int = ones(1,nVars);

% setup SMAC objects
smacremote = py.pysmac.smacremote.SMACRemote();
smacrunner = py.pysmac.smacrunner.SMACRunner(...
				x0, xmin, xmax, ...
				x0_int, xmin_int, xmax_int, ...
				py.dict({}), int32(smacremote.port), ...
				int32(nIter), int32(seed), ...
				int32(cutoff_time), ...
				int32(rf_num_trees), ...
				logical(rf_bootstrap), ...
				int32(int_percentage));


% initialize counter and current opt_soln
counter  = 0;
opt_x    = x0_int;
opt_obj  = Inf;

% Run loop to update current solution
while ~smacrunner.is_finished()

	% Update counter
	counter = counter + 1;
	smac_iter = tic;

	% check if new parameters are available 
	try
		params = smacremote.next();   
	catch
		continue
	end

	% extract updated solution vector
	params = struct(smacremote.get_next_parameters());    
    x_new = double(py.array.array('d',params.x_int));

    % evaluate objective
    y_new = objective(x_new);    

    % update optimal solution
    if y_new < opt_obj
        opt_x    = x_new;
        opt_obj = y_new;
    end

    % report solution back to SMAC instance
    runtime = toc(smac_iter);
    smacremote.report_performance(y_new, runtime);
    
    % Save models, model_obj, and runtime
    model_iter(counter,:) = opt_x;
    obj_iter(counter,1)   = opt_obj;
    time_iter(counter,1)  = runtime;

end

% save outputs
output = struct;
output.objVals  = obj_iter; 
output.optModel = model_iter;
output.runTime  = time_iter;

end