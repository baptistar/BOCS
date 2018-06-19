% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function Q = rand_ising_grid(n_vars)
% RAND_ISING: Function generates a random interaction matrix
% for an Ising Model on a grid graph with n_vars total variables

% Check that n_side is an integer
n_side = sqrt(n_vars);
if floor(n_side) ~= n_side
	error('Number of nodes is not square')
end

% Connect nodes horizontally
for i=1:n_side
	for j=1:n_side-1
		
		% Determine node idx
		node = (i-1)*n_side + j;

		Q(node,node+1) = 4.95*rand() + 0.05;%0.95*rand() + 0.05;
		Q(node+1,node) = Q(node,node+1);

	end
end

% Connect nodes vertically
for i=1:n_side-1
	for j=1:n_side
		
		% Determine node idx
		node = (i-1)*n_side + j;

		Q(node,node+n_side) = 4.95*rand() + 0.05;%0.95*rand() + 0.05;
		Q(node+n_side,node) = Q(node,node+n_side);

	end
end

% Apply random sign flips to Q
rand_sign = tril((rand(n_vars,n_vars) > 0.5)*2 - 1,-1);
rand_sign = rand_sign + rand_sign';
Q = rand_sign.*Q;

end
