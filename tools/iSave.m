% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function iSave(fname, rnd, sa, bo, ols, smc, smac, bayes, mle, hs, inputs_t)
	save(fname, 'rnd', 'sa', 'bo', 'ols', 'smc', 'smac', 'bayes', 'mle', 'hs', 'inputs_t')
end
