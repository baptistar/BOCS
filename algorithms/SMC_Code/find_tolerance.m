% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function particle_system = find_tolerance(particle_system)
% FIND_TOLERANCE: Function finds the tolerance to use for ABC in the SMC 
% algorithm based on the effective sample size.

% Extract function inputs
alpha     = particle_system.alpha_star;
rho       = particle_system.rho;
old_wts   = particle_system.weights;

% Declare parameters for iniital upper and lower bound
lower_bd = rho;
upper_bd = 5*rho;
epsilon  = rho/1e3;%min(rho/10,1e-4);

% Compute old ESS
old_ess  = eff_sample_size(old_wts);

% Declare desired ESS
ess_star = alpha*old_ess;

% Declare initial conditon for rho_t
rho_t = (lower_bd + upper_bd)/2;

% Perform bisection search
count_temp = 0;
while (abs(upper_bd - lower_bd) >= epsilon)

    % Compute weights with the new tolerance
    ps_weights = importance_weights(particle_system, rho_t);
    
    % If ESS is too high, increase rho
    if eff_sample_size(ps_weights) > ess_star
        lower_bd = rho_t;
        rho_t    = (rho_t + upper_bd)/2;
    % IF ESS is too low, decrease rho 
    else
        upper_bd = rho_t;
        rho_t    = (rho_t + lower_bd)/2; 
    end
   
    count_temp = count_temp + 1;
    if count_temp > 500
        break
    end
    
end

% If weights are NaN, decrease rho until it is nonzero
while all(isnan(ps_weights))
    rho_t = rho_t - epsilon;
    ps_weights = importance_weights(particle_system, rho_t);
end

% Assign weights corresponding to the final rho_t
particle_system.weights = ps_weights;
particle_system.rho     = rho_t;

end
