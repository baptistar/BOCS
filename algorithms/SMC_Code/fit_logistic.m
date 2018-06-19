% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function particle_system = fit_logistic(particle_system)
% FIT_LOGISTIC: Function fits a weighted logistic regression to the data to
% find an optimal proposal for moving particles

% Extract variables
models  = particle_system.models;
weights = particle_system.weights;
A = particle_system.logistic_A;

% Assign a delta tolerance
delta = 0.01;

% Assign epsilon for nugget
epsilon = 0.1;

% Assign tolerances for logistic regression
eps_mean = 0.02;
eps_corr = 0.075;

% Assign nugget for minimum marginal probability
min_marg_prob = 1e-10;

% Determine the number of samples and dimensions
[n_samples, n_vars] = size(models);

% Set maximum number of iterations
iter_max = 50;

% Compute the statistics of the dataset
[s_mean, ~, s_corr] = samp_stats(models, weights);

% Find optimal A for each dimension
for i=1:n_vars
    
    % Check if sample mean is outside the intervsl (eps_mean, 1 - eps_mean)
    if (s_mean(i) <= eps_mean || s_mean(i) >= (1 - eps_mean))
        
        % Add nugget to mean
        s_mean_adj = s_mean(i)*(1 - min_marg_prob) + min_marg_prob*0.5;
        
        % Compute components of A
        A(i,i) = log(s_mean_adj) - log(1 - s_mean_adj);
        A(i,1:i-1) = 0;
        
        continue
        
    end
    
    % Determine the regressors based on sample correlation
    L_i = abs(s_corr(i,1:i-1)) > eps_corr;

    % Extract columns from samples
    Z = [models(:,L_i), ones(n_samples,1)];
    y = models(:,i);
    
    % Extract row from A (transform into column vector)
    a_old = [A(i,L_i)'; A(i,i)];
    
    % Declare initial error
    error = Inf;
    
    % Declare counter
    counter = 0;
    
    % Optimize while the error is large
    while((error > delta) && counter < iter_max)
        
        % Declare vectors to store evaluations of the logistic function
        p_log = zeros(n_samples,1);
        q_log = zeros(n_samples,1);
        
        for k=1:n_samples
            
            % Determine p_log and q_log for each sample
            p_log(k) = 1/(1 + exp(-Z(k,:)*a_old)); 
            q_log(k) = p_log(k)*(1-p_log(k));
            
        end
        
        n_nugget = size(Z,2);
        
        % Determine new row for A
        a_new = a_old + (Z'*diag(weights)*diag(q_log)*Z + epsilon*eye(n_nugget))\...
                        (Z'*diag(weights)*(y - p_log) - epsilon*a_old);

        % Determine error and assign a_old
        error = norm(a_new - a_old,Inf);
        a_old = a_new;
              
        % Update counte
        counter = counter + 1;
        
    end
    
    % Assign a_new to A (row vector)
    A(i,L_i) = a_new(1:end-1)';
    A(i,i)   = a_new(end);
    
end

% Save and return particle_system
particle_system.logistic_A = A;

end