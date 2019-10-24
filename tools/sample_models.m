% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function models = sample_models(n_models, n_vars, values)
% SAMPLE_MODELS: Sample candidate model vectors with categorical values
%
% Inputs: n_models - number of vectors to return
%         n_vars - number of variables for each vector
%         values - {n_vars x 1} cell - numeric variables for each dimension

% check values and default to binary if not specified
if nargin < 3
	values = repmat({[0,1]},n_vars,1);
end

% check if n_vars and values have the same dimension
if length(values) ~= n_vars
    error('Number of variables does not match size of values cell')
end

% Generate matrix of zeros to store models
models = zeros(n_models, n_vars);

% Construct model vectors by sampling each dimension independently
% and with replacement
for j=1:n_vars
	models(:,j) = randsample(values{j}, n_models, true);
end

% -- END OF FILE --