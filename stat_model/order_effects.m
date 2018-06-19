% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function x_allpairs = order_effects(x_vals, ord_t)
% order_effects: Function computes data matrix for all coupling
% orders to be added into linear regression model.

% Find number of variables
[n_samp, n_vars] = size(x_vals);

% Generate matrix to store results
x_allpairs = x_vals;

for ord_i=2:ord_t

	% generate all combinations of indices
	offdProd = combnk(1:n_vars,ord_i);
	diagProd = repmat(1:n_vars,ord_i,1)';
	cartProd = [offdProd; diagProd];

	x_comb = zeros(n_samp, size(cartProd,1), ord_i);
	for j=1:ord_i
		x_comb(:,:,j) = x_vals(:,cartProd(:,j));
	end
	x_allpairs = [x_allpairs, prod(x_comb,3)];

end

end