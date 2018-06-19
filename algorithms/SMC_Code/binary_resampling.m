% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function [x_bin_new, x_new_idx] = binary_resampling(binary_models, weights)
% BINARY_RESAMPLING: Function re-samples binary particles based on the 
% current weights using systematic resampling.

% Determine the size of x_bin
[n_samples,n_var] = size(binary_models);

% Declare a vector to store new samples and indices
x_bin_new  = zeros(n_samples, n_var);
x_new_idx  = zeros(n_samples, 1);

% Initialize system (u = U[0,1] & v = weights*n_samples & c = v(1))
u = rand;
v = weights*n_samples;
c = v(1);

% Declare iteration counter
counter = 1;

% Resample each particle
for i=1:n_samples
    
    % Update counter and c value
    while c < u
        counter = counter+1;
        c       = c + v(counter);
    end
    
    % Assign new binary vector and probability
    x_bin_new(i,:) = binary_models(counter,:);
    x_new_idx(i)   = counter;
    
    % Update random value
    u = u+1;
    
end

end