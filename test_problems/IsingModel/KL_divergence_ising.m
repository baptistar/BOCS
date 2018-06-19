% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function KL = KL_divergence_ising(Theta_P, moments, x)
% KL_divergence_ising: Function evaluates the KL divergence objective
% for Ising Models

n_vars = size(Theta_P,1);

% Generate all binary vectors
bin_vals = dec2bin(0:2^n_vars-1)-'0';
bin_vals(bin_vals == 0) = -1;
n_vectors = size(bin_vals,1);

% Compute normalizing constant for P
P_vals = zeros(n_vectors,1);
for i=1:n_vectors
	P_vals(i) = exp(bin_vals(i,:)*Theta_P*bin_vals(i,:)');
end
Zp = sum(P_vals);

% Run computation for each x
n_xvals = size(x,1);
KL = zeros(n_xvals,1);

for j=1:n_xvals
    
    % Apply elementwise masking to Theta
    Theta_Q = tril(Theta_P,-1);
    nnz_Q   = find(Theta_Q);
    Theta_Q(nnz_Q) = Theta_Q(nnz_Q).*(x(j,:))';
    Theta_Q = Theta_Q + Theta_Q';

    % Compute normalizing constant for Q
    Q_vals = zeros(n_vectors,1);
    for i=1:n_vectors
        Q_vals(i) = exp(bin_vals(i,:)*Theta_Q*bin_vals(i,:)');
    end
    Zq = sum(Q_vals);

    % compute KL
    KL(j) = sum(sum((Theta_P - Theta_Q).*moments)) + log(Zq) - log(Zp);

end

end