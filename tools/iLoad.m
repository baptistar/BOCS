% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [rnd, sa, bo, ols, smc, smac, bayes, mle, hs, inputs_t] = iLoad(file_name);
	load(file_name, 'rnd', 'sa', 'bo', 'ols', 'smc', 'smac', 'bayes', 'mle', 'hs', 'inputs_t');
end
