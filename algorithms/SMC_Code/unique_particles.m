% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function n_particles = unique_particles(binary_particles)
% UNIQUE_PARTICLES: Function determines the number of unique particles 
% in the current system.

% Determine the number of total particles
[n_models, ~] = size(binary_particles);

% Find unique particles
unique_particles = unique(binary_particles,'rows');

% Find n_particles
n_particles = size(unique_particles,1)/n_models;
