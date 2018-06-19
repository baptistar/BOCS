% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [bayes_err, mle_err, hs_err] = compare_models(inputs)
% CROSS_VALIDATE: Function performs cross-validation of the trained model
% to determine if it 

% Extract test samples
x_test = inputs.x_test;
y_test = inputs.y_test;
n_test = size(inputs.x_test,1);

% Extract settings for training
nVars_ = inputs.n_vars;
order_ = inputs.order;

% Extract parameters for Horseshoe runs
nHSavg = inputs.nHSavg;

% Setup statistical models
LR_bayes = LinReg(nVars_, order_, 'bayes');
LR_mle   = LinReg(nVars_, order_, 'mle');
LR_hs    = LinReg(nVars_, order_, 'horseshoe');

% Compute model x values
x_all = [ones(n_test,1), order_effects(x_test, inputs.order)];

% compute statistics of Bayes estimator
LR_bayes   = LR_bayes.train(inputs);
bayes_mean = x_all*LR_bayes.alphaMean;
bayes_var  = LR_bayes.b/LR_bayes.a*(2*LR_bayes.a)/(2*LR_bayes.a - 2)*...
			 (eye(n_test) + x_all*LR_bayes.alphaCov*x_all');
bayes_std  = sqrt(diag(bayes_var));

% compute statistics of MLE estimator
LR_mle   = LR_mle.train(inputs);
mle_mean = x_all*LR_mle.alpha_mle;

% compute statistics of horseshoe estimator
y_pred_hs = zeros(n_test, nHSavg);
for i=1:nHSavg
	LR_hs = LR_hs.train(inputs);
	y_pred_hs(:,i) = x_all*LR_hs.sampleCoeff();
end
hs_mean = mean(y_pred_hs,2);
hs_std  = std(y_pred_hs,[],2);

% compute mean errors
bayes_err = abs(bayes_mean - y_test);
mle_err   = abs(mle_mean - y_test);
hs_err    = abs(hs_mean - y_test);

end