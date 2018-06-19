% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function run_cases(inputs_all, lambda_vals, test_name, n_proc)
% RUN_CASES: Functions runs the discrete optimization algorithms
% on all test cases specified in inputs_all with the lambda values
% prescribed in lambda_vals vector. The results are saved in the
% results/test_name folder.

% Find number of tests
n_test = length(inputs_all);

%% Parallel Problem setup

parpool(n_proc);
parfor t=1:n_test    

    % Set test inputs struct
    inputs_t = inputs_all{t};

    % Declare cells to store results
    rnd  = cell(length(lambda_vals));
    sa   = cell(length(lambda_vals));
    bo   = cell(length(lambda_vals));
    ols  = cell(length(lambda_vals));
    smc  = cell(length(lambda_vals));
    smac = cell(length(lambda_vals));

    bayes = struct;
    bayes.stSA1 = cell(length(lambda_vals));
    bayes.stSA2 = cell(length(lambda_vals));
    bayes.stSA3 = cell(length(lambda_vals));
    bayes.sdp   = cell(length(lambda_vals));

    mle = struct;
    mle.stSA1   = cell(length(lambda_vals));
    mle.stSA2   = cell(length(lambda_vals));
    mle.stSA3   = cell(length(lambda_vals));
    mle.sdp     = cell(length(lambda_vals));

    hs = struct;
    hs.stSA1    = cell(length(lambda_vals));
    hs.stSA2    = cell(length(lambda_vals));
    hs.stSA3    = cell(length(lambda_vals));
    hs.sdp      = cell(length(lambda_vals));

    %% Run optimization
    for l=1:length(lambda_vals)
    
        % Define objective function with penalty term
        inputs_t.lambda = lambda_vals(l);
        penalty   = @(x) inputs_t.lambda*inputs_t.reg_term(x);
        objective = @(x) inputs_t.model(x) + penalty(x);

        fprintf('--------------------------------------------\n')
        fprintf('Test = %d/%d, Lambda = %f\n\n', t, n_test, lambda_vals(l));

        % Run different ML optimization algorithms
        rnd{l} = random_samp(objective, inputs_t); 
        fprintf('Random - Runtime: %f\n', sum(rnd{l}.runTime));

        sa{l}  = simulated_annealing(objective, inputs_t);
        fprintf('SA - Runtime = %f\n', sum(sa{l}.runTime));

        bo{l}  = bayes_opt(objective, inputs_t);
        fprintf('BO - Runtime = %f\n', sum(bo{l}.runTime));

        ols{l} = local_search(objective, inputs_t);
        fprintf('OLS - Runtime = %f\n', sum(ols{l}.runTime));

        smc{l} = binary_smc(objective, inputs_t);
        fprintf('SMC - Runtime = %f\n', sum(smc{l}.runTime));

        smac{l} = run_smac(objective, inputs_t);
        fprintf('SMAC - Runtime = %f\n', sum(smac{l}.runTime));

        % Run BOCS with Bayesian model
        inputs_t.estimator = 'bayes';

        bayes.stSA1{l} = BOCS(inputs_t.model, penalty, inputs_t, 1, 'SA');
        fprintf('Bayes.SA1 - Runtime = %f\n', sum(bayes.stSA1{l}.runTime));

        bayes.stSA2{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'SA');
        fprintf('Bayes.SA2 - Runtime = %f\n', sum(bayes.stSA2{l}.runTime));

        bayes.stSA3{l} = BOCS(inputs_t.model, penalty, inputs_t, 3, 'SA');
        fprintf('Bayes.SA3 - Runtime = %f\n', sum(bayes.stSA3{l}.runTime));

        bayes.sdp{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'sdp');
        fprintf('Bayes.SDP - Runtime = %f\n', sum(bayes.sdp{l}.runTime));

        % Run BOCS with MLE model
        inputs_t.estimator = 'mle';

        mle.stSA1{l} = BOCS(inputs_t.model, penalty, inputs_t, 1, 'SA');
        fprintf('MLE.SA1 - Runtime = %f\n', sum(mle.stSA1{l}.runTime));

        mle.stSA2{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'SA');
        fprintf('MLE.SA2 - Runtime = %f\n', sum(mle.stSA2{l}.runTime));

        mle.stSA3{l} = BOCS(inputs_t.model, penalty, inputs_t, 3, 'SA');
        fprintf('MLE.SA3 - Runtime = %f\n', sum(mle.stSA3{l}.runTime));

        mle.sdp{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'sdp');
        fprintf('MLE.SDP - Runtime = %f\n', sum(mle.sdp{l}.runTime));

        % Run BOCS with Horseshoe model
        inputs_t.estimator = 'horseshoe';

        hs.stSA1{l} = BOCS(inputs_t.model, penalty, inputs_t, 1, 'SA');
        fprintf('HS.SA1 - Runtime = %f\n', sum(hs.stSA1{l}.runTime));

        hs.stSA2{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'SA');
        fprintf('HS.SA2 - Runtime = %f\n', sum(hs.stSA2{l}.runTime));

        hs.stSA3{l} = BOCS(inputs_t.model, penalty, inputs_t, 3, 'SA');
        fprintf('HS.SA3 - Runtime = %f\n', sum(hs.stSA3{l}.runTime));

        hs.sdp{l} = BOCS(inputs_t.model, penalty, inputs_t, 2, 'sdp');
        fprintf('HS.SDP - Runtime = %f\n', sum(hs.sdp{l}.runTime));

        % Save results
        iSave(sprintf(['../results/' test_name '/test%d.mat'], t), ...
            rnd, sa, bo, ols, smc, smac, bayes, mle, hs, inputs_t);

    end
end

delete(gcp('nocreate'))