% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function out = bayes_obj(objective, x)
	out = objective(double(table2array(x)));
end
