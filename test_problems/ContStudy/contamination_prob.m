% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function out = contamination_prob(x, n_samples, seed)

% Declare gamma factor (Lagrange constants)
gamma = 1;

% Find total number of input samples
num_inputs = size(x,1);
out = zeros(num_inputs,1);

for i=1:num_inputs
    
    % Run contamination study
    [cost, ~, ~, ~, constraint, ~, ~, ~] = Contamination(x(i,:)', n_samples, seed);

    % Compute total output
    out(i) = cost - sum(gamma*constraint);

end
