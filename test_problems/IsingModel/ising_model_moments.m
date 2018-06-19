% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function ising_mom = ising_model_moments(Q)

n_vars = size(Q,1);

% Generate all binary vectors
bin_vals = dec2bin(0:2^n_vars-1)-'0';
bin_vals(bin_vals == 0) = -1;
n_vectors = size(bin_vals,1);

% Compute values of PDF
pdf_vals = zeros(n_vectors,1);
for i=1:n_vectors
	pdf_vals(i) = exp(bin_vals(i,:)*Q*bin_vals(i,:)');
end

% Compute normalizing constant
norm_const = sum(pdf_vals);

% Generate matrix to store moments
ising_mom = zeros(n_vars,n_vars);

% Compute second moment for each pair of values
for i=1:n_vars
	for j=1:n_vars
		bin_pair = bin_vals(:,i).*bin_vals(:,j);
		ising_mom(i,j) = sum(bin_pair.*pdf_vals)/norm_const;
	end
end