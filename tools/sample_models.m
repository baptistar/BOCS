% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [binary_models] = sample_models(n_models, n_vars)
% SAMPLE_MODELS: Function samples the binary models to
% generate observations to train the statistical model

% Generate matrix of zeros with ones along diagonals
binary_models = zeros(n_models, n_vars);

% Sample model indices
model_num = randi(2^n_vars, n_models,1);

% Construct each binary model vector
for i=1:n_models
	model = dec2bin(model_num(i)-1) - '0';
	binary_models(i,:) = [zeros(1,n_vars - length(model)), model];    
end

end