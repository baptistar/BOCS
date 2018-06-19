% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function ess = eff_sample_size(weights)
% EFF_SAMPLE_SIZE: Function computes the effective sample size for a
% collection of particles based on the weights.

% Check if values are NaN
if all(isnan(weights))
    ess = 0;
    
% Compute standard ESS
else
    ess = 1/(sum(weights.^2));
    
end

end