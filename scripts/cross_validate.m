%
% Bayesian Optimization of Combinatorial Structures
%
% Copyright (C) 2018 R. Baptista & M. Poloczek
%
% BOCS is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% BOCS is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License 
% along with BOCS.  If not, see <http://www.gnu.org/licenses/>.
%
% Copyright (C) 2018 MIT & University of Arizona
% Authors: Ricardo Baptista & Matthias Poloczek
% E-mails: rsb@mit.edu & poloczek@email.arizona.edu
%

% Script runs validation script for each set of test functions:
%   - Quadratic programming problem
%   - Ising model sparsification problem
%   - Contamination control problem

clear; close all; clc
addpath(genpath('../stat_model'))
addpath(genpath('../test_problems'))
addpath(genpath('../tools'))
addpath(genpath('../plotting'))
addpath(genpath('../validation'))

%% SETUP FIXED PARAMETERS

% Setup fixed parameters
order  = 1:3;
n_proc = 1;

% Initial sample size for statistical models
n_train = 20:10:150;
n_func  = 10;
n_test  = 50;

% Variance prior parameters (Inverse Gamma)
aPr    = 2;
bPr    = 1;

% Number of samples to average sparse linear regression models
nHSavg = 10;

for ord=1:length(order)

    %% QUADRATIC TEST PROBLEM

    % problem parameters
    n_vars     = 10;
    alpha_vect = logspace(0,2,3);
    test_name  = 'quad';

    % Declare cell to store all inputs
    inputs_all = cell(n_func*length(alpha_vect),1);

    for a=1:length(alpha_vect)
    	for t=1:n_func

    		idx = (a-1)*n_func+t;

    		% Set inputs struct for each problem
    		inputs_all{idx} = struct;
    		inputs_all{idx}.n_vars  = n_vars;
    		inputs_all{idx}.order   = order(ord);

    		% Set parameters for estimators
    		inputs_all{idx}.aPr     = aPr;
    		inputs_all{idx}.bPr     = bPr;
    		inputs_all{idx}.nHSavg  = nHSavg;

    		% Generate random test function
    		Qa = quad_mat(n_vars, alpha_vect(a));
    		inputs_all{idx}.model = @(x) diag(x*Qa*x');
    		inputs_all{idx}.name  = test_name;
    	
    	end
    end

    validationPlots(inputs_all, n_train, n_test, n_proc);

    %% CONTAMINATION TEST PROBLEM

    % problem parameters
    n_vars    = 20;
    mcSamps   = 1e3;
    test_name = 'contamination';

    % Declare cell to store all inputs
    inputs_all = cell(n_func,1);

    for t=1:n_func

        % Set inputs struct for each problem
        inputs_all{t} = struct;
        inputs_all{t}.n_vars  = n_vars;
        inputs_all{t}.order   = order(ord);
        
        % Set parameters for estimators
        inputs_all{t}.aPr     = aPr;
        inputs_all{t}.bPr     = bPr;
        inputs_all{t}.nHSavg  = nHSavg;

        % Generate random graphical model
        seed = randi(10000,1);
        inputs_all{t}.model = @(x) contamination_prob(x, mcSamps, seed);
    	inputs_all{t}.name  = test_name;

    end

    validationPlots(inputs_all, n_train, n_test, n_proc);

    %% ISING MODEL TEST PROBLEM

    % problem parameters
    n_nodes = 16;
    n_vars  = 24;
    test_name = 'ising';

    % Declare cell to store all inputs
    inputs_all = cell(n_func,1);

    for t=1:n_func

        % Set inputs struct for each problem
        inputs_all{t} = struct;
        inputs_all{t}.n_vars  = n_vars;
        inputs_all{t}.order   = order(ord);
        
        % Set parameters for estimators
        inputs_all{t}.aPr     = aPr;
        inputs_all{t}.bPr     = bPr;
        inputs_all{t}.nHSavg  = nHSavg;

        % Generate random graphical model
        Theta   = rand_ising_grid(n_nodes);
        Moments = ising_model_moments(Theta);
        inputs_all{t}.model = @(x) KL_divergence_ising(Theta, Moments, x);
    	inputs_all{t}.name  = test_name;

    end

    validationPlots(inputs_all, n_train, n_test, n_proc);

end

% -- END OF FILE --
