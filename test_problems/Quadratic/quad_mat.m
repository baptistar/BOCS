% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function Qa = quad_mat(n_vars, alpha)

	% compute decay function
	K = @(s,t) exp(-1*(s-t)^2/alpha);
	decay = zeros(n_vars,n_vars);

	for i=1:n_vars
		for j=1:n_vars
			decay(i,j) = K(i,j);
		end
	end

	% Generate random quadratic model
	% and apply exponential decay to Q
	Q = randn(n_vars, n_vars);
	Qa = Q.*decay;

end