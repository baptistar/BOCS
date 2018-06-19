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

% Script runs discrete optimization algorithms for the
% contamination control simulation problem. The results are 
% compared for different values of the \lambda tuning 
% parameter.

clear; close all; clc
addpath(genpath('../algorithms'))
addpath(genpath('../stat_model'))
addpath(genpath('../test_problems/ContStudy'))
addpath(genpath('../tools'))

%% Setup parameters

% Setup fixed parameters
n_vars  = 30;
n_proc  = 1;
test_name = 'contamination';

% Number of runs and optimization iterations
n_func     = 10;
n_runs     = 10;
n_init     = 20;
evalBudget = 270;

% problem parameters (Monte Carlo samples)
mcSamps = 1e2;

% Variance prior parameters (Inverse Gamma)
aPr    = 2;
bPr    = 1;

% Regularization parameters
lambda_vals = [0, 1e-4, 1e-2, 1];
lambda_str  = {'0', '1em4', '1em2', '1'};

% Set additive regularization function
reg_term = @(x) sum(x,2);

%% Generate Test Cases

% setup objective functions
inputs_all = cell(n_func, n_runs);

for t1=1:n_func

    fprintf('Setting up test function %d\n', t1);

    % Generate random case study
    seed = randi(10000,1);

    for t2=1:n_runs

        % Set inputs struct for each problem
        inputs_all{t1,t2} = struct;
        inputs_all{t1,t2}.n_vars      = n_vars;

        % Save other definitions
        inputs_all{t1,t2}.evalBudget  = evalBudget;
        inputs_all{t1,t2}.n_runs      = n_runs;
        inputs_all{t1,t2}.n_init      = n_init;
        inputs_all{t1,t2}.lambda_vals = lambda_vals;

        % Set priors for estimator
        inputs_all{t1,t2}.aPr         = aPr;
        inputs_all{t1,t2}.bPr         = bPr;

        % Save objective function and regularization term
        inputs_all{t1,t2}.model = @(x) contamination_prob(x, mcSamps, seed);
        inputs_all{t1,t2}.reg_term = @(x) reg_term(x);

        % Generate initial samples for statistical models
        inputs_all{t1,t2}.x_vals = sample_models(n_init, n_vars);
        inputs_all{t1,t2}.y_vals = inputs_all{t1,t2}.model(inputs_all{t1,t2}.x_vals);

    end
end

inputs_all = reshape(inputs_all, n_func*n_runs, 1);

% Make folder
mkdir(['../results/' test_name])

% Save test cases
save(['../results/' test_name '/all_tests'])

% Run cases
run_cases(inputs_all, lambda_vals, test_name, n_proc);

% -- END OF FILE --