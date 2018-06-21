# Bayesian Optimization of Combinatorial Structures (BOCS)

## What is the BOCS algorithm?

The BOCS algorithm is a method for finding a global minimizer of an expensive-to-evaluate black-box function that is defined over discrete inputs given a finite budget of function evaluations. The algorithm combines an adaptive generative model and semidefinite programming techniques for scalability of the acquisition function to large combinatorial domains.

## Authors

Ricardo Baptista (MIT) and Matthias Poloczek (University of Arizona)

E-mails: rsb@mit.edu, poloczek@email.arizona.edu

## Installation

The BOCS algorithm is implemented in MATLAB and only requires the [CVX package](http://cvxr.com/cvx/) to be available in the local path for performing convex optimization. The scripts for comparing to other discrete optimization methods require an installation of [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/), the python wrapper [pySMAC](https://github.com/tdomhan/pysmac) and the [bayesopt](https://www.mathworks.com/help/stats/bayesopt.html) function in MATLAB for Bayesian Optimization with the Expected Improvement acquisition function. 

## Example running BOCS on a benchmark problem

We provide an example for running the BOCS algorithm the Ising model sparsification benchmark problem on a 9-node grid graph with a budget of 100 sample evaluations. The code can also be run from MATLAB using the file `scripts/example_ising.m`.

The script first defines the input parameters in the `inputs` struct. These include the number of discrete variables (i.e., edges in the grid graph) `n_vars`, the sample evaluation budget `evalBudget`, the number of samples initially used to build the statistical model `n_init`, the lambda parameter `lambda`, and the horseshoe `estimator` used for the regression model. Other options for `estimator` are bayes and mle.

```Matlab

inputs = struct;
inputs.n_vars     = 12;
inputs.evalBudget = 100;
inputs.n_init     = 20;
inputs.lambda     = 1e-4;
inputs.estimator  = 'horseshoe';

```

We randomly instantiate the edge weights of the Ising model and define the objective function based on the KL divergence in `inputs.model`. We add an l_1 penalty scaled by the lambda parameter in `inputs.penalty`. 

```Matlab

% Generate random graphical model
Theta   = rand_ising_grid(9);
Moments = ising_model_moments(Theta);

% Save objective function and regularization term
inputs.model    = @(x) KL_divergence_ising(Theta, Moments, x);
inputs.penalty  = @(x) inputs.lambda*sum(x,2);

```

We sample the initial values for the statistical model and evaluate the objective function at these samples. Using these inputs, we run the BOCS-SA and BOCS-SDP algorithms with the order 2 regression model.

```Matlab

% Generate initial samples for statistical models
inputs.x_vals   = sample_models(inputs.n_init, inputs.n_vars);
inputs.y_vals   = inputs.model(inputs.x_vals);

% Run BOCS-SDP and BOCS-SA (order 2)
B_SA  = BOCS(inputs.model, inputs.penalty, inputs, 2, 'SA');
B_SDP = BOCS(inputs.model, inputs.penalty, inputs, 2, 'sdp');

```

## Comparison to other discrete optimization algorithms

To compare BOCS to other algorithms, we provided files in the scripts folder that run all cases for the quadratic programming problem, the Ising model sparsification problem, and the contamination control problem. The algorithms that are compared include:

- RS: Random search [Bergstra & Bengio, 2012](http://www.jmlr.org/papers/v13/bergstra12a.html) 
- SA: Simulated annealing [Spears, 1993](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.52.6986) [Pankratov & Borodin, 2010](https://link.springer.com/chapter/10.1007/978-3-642-14186-7_19)
- EI: Expected improvement [Jones et al., 2008](https://link.springer.com/article/10.1023/A:1008306431147) 
- OLS: Local search [Khanna et al., 1998](https://epubs.siam.org/doi/abs/10.1137/S0097539795286612)
- PS: Sequential Monte Carlo particle search [Schäfer, 2012](https://dl.acm.org/citation.cfm?id=2414424)
- SMAC [Hutter et al., 2011](https://dl.acm.org/citation.cfm?id=2177404)
- BOCS-SA, BOCS-SDP with Bayesian linear regression model
- BOCS-SA, BOCS-SDP with MLE model
- BOCS-SA, BOCS-SDP with sparse linear regression model

To run these examples, we provided three files in the scripts folder named `opt_runs_quad`, `opt_runs_ising` and `opt_runs_cont` for the benchmark problems. In each of the files, the `n_func` parameter defines multiple random instances of the test function and the `n_runs` parameter defines the number of repeated evaluations of each algorithm starting from different initial datasets. The `n_proc` parameter can be used to specify the number of available processors to parallelize the execution of these tests. Lastly, running `scripts/create_plots` will plot the average output of these tests and present statistics of the solutions found by all algorithms, while running 'scripts/cross_validate` followed by `scripts/plot_crossval` can be used to validate the statistical models for each benchmark problem.
