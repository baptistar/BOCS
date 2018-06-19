% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [model, obj] = sdp_relaxation(alpha_vect, inputs)
% SDP_Relaxation: Function runs simulated annealing algorithm for 
% optimizing binary functions. The function returns optimum models and min 
% objective values found at each iteration

% Extract n_vars
n_vars = inputs.n_vars;

% Extract vector of coefficients
b = alpha_vect(2:n_vars+1) + inputs.lambda;
a = alpha_vect(n_vars+2:end);

% get indices for quadratic terms
offdProd = combnk(1:n_vars,2);
diagProd = repmat(1:n_vars,2,1)';
idx_prod = [offdProd; diagProd];
n_idx = size(idx_prod,1);

% check number of coefficients
if length(a) ~= n_idx
    error('Number of Coefficients does not match indices!')
end

% Convert a to matrix form
A = zeros(n_vars,n_vars);
for i=1:n_idx
	A(idx_prod(i,1),idx_prod(i,2)) = a(i)/2;
	A(idx_prod(i,2),idx_prod(i,1)) = a(i)/2;
end

% Convert to standard form
bt = b/2 + A*ones(n_vars,1)/2;
At = [A/4, bt/2; bt'/2, 0];

% Run SDP relaxation
cvx_begin sdp quiet
    variable X(n_vars+1,n_vars+1) symmetric
    minimize trace(At*X)
    subject to
        X>=0
        diag(X)==ones(n_vars+1,1)
cvx_end

% Extract vectors and compute Cholesky
L = chol(X);

% Repeat rounding for different vectors
n_rand_vector = 100;

model_vect = zeros(n_vars,n_rand_vector);
obj_vect   = zeros(1,n_rand_vector);

for kk=1:n_rand_vector

     % Generate a random cutting plane vector (uniformly distributed on the
     % unit sphere - normalized vector)
     r = randn(n_vars+1,1); r = r/norm(r);
     y_soln = sign(L'*r);
     
     % convert solution to original domain and assign to output vector
     model_vect(:,kk) = (y_soln(1:n_vars)+1)/2;
     obj_vect(kk) = model_vect(:,kk)'*A*model_vect(:,kk) + b'*model_vect(:,kk);
     
%     % Declare vectors to store z_i and y_i
%     z = zeros(n_vars+1,1);
%     y = zeros(n_vars+1,1);
% 
%     T = sqrt(4*log(n_vars+1));
%     for i=1:n_vars+1
%         z(i) = L(:,i)'*r/T;
%         if abs(z(i)) > 1
%             y(i) = z(i)/abs(z(i));
%         else
%             y(i) = z(i);
%         end
%     end
% 
%     % round solution randomly (flip sign with some prob)
%     x = ones(n_vars+1,1);
%     for i=1:n_vars+1
%         if rand < (1 - y(i))/2
%             x(i) = -1;
%         end
%     end
    
end

% Find optimal rounded solution
[~,opt_idx] = min(obj_vect);
model = model_vect(:,opt_idx)';
obj   = obj_vect(opt_idx);

end