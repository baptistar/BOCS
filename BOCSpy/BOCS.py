# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np
import cvxpy as cvx
from itertools import combinations
from LinReg import LinReg
from sample_models import sample_models

def BOCS(inputs, order, acquisitionFn):
	# BOCS: Function runs binary optimization using simulated annealing on
	# the model drawn from the distribution over beta parameters
	#
	# Inputs: inputs (dictionary) specifying the parameters:
	#			- n_vars: number of variables 
	#			- evalBudget: max number of function evaluations
	#			- (x_vals, y_vals): initial samples
	# 			- model: target objective function f(x)
	#			- penalty: regularization term P(x)
	# 		   order (int): statistical model order
	# 		   aquisitionFn: 'SA' or 'SDP'

	# Set the number of SA reruns
	SA_reruns = 5

	# Extract inputs
	n_vars  = inputs['n_vars']
	model   = inputs['model']
	penalty = inputs['penalty']

	# Train initial statistical model
	LR = LinReg(n_vars, order)
	LR.train(inputs)

	# Find number of iterations based on total budget
	n_init = inputs['x_vals'].shape[0]
	n_iter = inputs['evalBudget'] - n_init

	# Declare vector to store results
	model_iter = np.zeros((n_iter, n_vars))
	obj_iter   = np.zeros(n_iter)

	for t in range(n_iter):

		# Draw alpha vector
		alpha_t = LR.alpha

		# Run SA optimization
		if acquisitionFn == 'SA':

			# Setup statistical model objective for SA
			stat_model = lambda x: LR.surrogate_model(x, alpha_t) + penalty(x)

			SA_model = np.zeros((SA_reruns, n_vars))
			SA_obj	 = np.zeros(SA_reruns)

			for j in range(SA_reruns):
				(optModel, objVals) = simulated_annealing(stat_model, inputs)
				SA_model[j,:] = optModel[-1,:]
				SA_obj[j]	  = objVals[-1]

			# Find optimal solution
			min_idx = np.argmin(SA_obj)
			x_new = SA_model[min_idx,:]

		# Run semidefinite relaxation for order 2 model with l1 loss
		elif order == 2 and acquisitionFn == 'SDP-l1':
			x_new, _ = sdp_relaxation(alpha_t, inputs)
		else:
			raise NotImplementedError

		# evaluate model objective at new evaluation point
		x_new = x_new.reshape((1,n_vars))
		y_new = model(x_new)

		# Update inputs dictionary
		inputs['x_vals'] = np.vstack((inputs['x_vals'], x_new))
		inputs['y_vals'] = np.hstack((inputs['y_vals'], y_new))
		inputs['init_cond'] = x_new

		# re-train linear model
		LR.train(inputs)

		# Save results for optimal model
		model_iter[t,:] = x_new
		obj_iter[t]		= y_new + penalty(x_new)

	return (model_iter, obj_iter)


def simulated_annealing(objective, inputs):
	# SIMULATED_ANNEALING: Function runs simulated annealing algorithm for 
	# optimizing binary functions. The function returns optimum models and min 
	# objective values found at each iteration

	# Extract inputs
	n_vars = inputs['n_vars']
	n_iter = inputs['evalBudget']

	# Declare vectors to save solutions
	model_iter = np.zeros((n_iter,n_vars))
	obj_iter   = np.zeros(n_iter)

	# Set initial temperature and cooling schedule
	T = 1.
	cool = lambda T: .8*T

	# Set initial condition and evaluate objective
	old_x   = sample_models(1,n_vars)
	old_obj = objective(old_x)

	# Set best_x and best_obj
	best_x   = old_x
	best_obj = old_obj

	# Run simulated annealing
	for t in range(n_iter):

		# Decrease T according to cooling schedule
		T = cool(T)

		# Find new sample
		flip_bit = np.random.randint(n_vars)
		new_x = old_x.copy()
		new_x[0,flip_bit] = 1. - new_x[0,flip_bit]

		# Evaluate objective function
		new_obj = objective(new_x)

		# Update current solution iterate
		if (new_obj < old_obj) or (np.random.rand() < np.exp( (old_obj - new_obj)/T )):
			old_x   = new_x
			old_obj = new_obj

		# Update best solution
		if new_obj < best_obj:
			best_x   = new_x
			best_obj = new_obj

		# save solution
		model_iter[t,:] = best_x
		obj_iter[t]		= best_obj

	return (model_iter, obj_iter)


def sdp_relaxation(alpha, inputs):
	# SDP_Relaxation: Function runs simulated annealing algorithm for 
	# optimizing binary functions. The function returns optimum models and min 
	# objective values found at each iteration

	# Extract n_vars
	n_vars = inputs['n_vars']

	# Extract vector of coefficients
	b = alpha[1:n_vars+1] + inputs['lambda']
	a = alpha[n_vars+1:]

	# get indices for quadratic terms
	idx_prod = np.array(list(combinations(np.arange(n_vars),2)))
	n_idx = idx_prod.shape[0]

	# check number of coefficients
	if a.size != n_idx:
	    raise ValueError('Number of Coefficients does not match indices!')

	# Convert a to matrix form
	A = np.zeros((n_vars,n_vars))
	for i in range(n_idx):
		A[idx_prod[i,0],idx_prod[i,1]] = a[i]/2.
		A[idx_prod[i,1],idx_prod[i,0]] = a[i]/2.

	# Convert to standard form
	bt = b/2. + np.dot(A,np.ones(n_vars))/2.
	bt = bt.reshape((n_vars,1))
	At = np.vstack((np.append(A/4., bt/2.,axis=1),np.append(bt.T,2.)))

	# Run SDP relaxation
	X = cvx.Variable((n_vars+1, n_vars+1), PSD=True)
	obj = cvx.Minimize(cvx.trace(cvx.matmul(At,X)))
	constraints = [cvx.diag(X) == np.ones(n_vars+1)]
	prob = cvx.Problem(obj, constraints)
	prob.solve(solver=cvx.CVXOPT)

	# Extract vectors and compute Cholesky
	# add small identity matrix is X.value is numerically not PSD
	try:
		L = np.linalg.cholesky(X.value)
	except:
		XpI = X.value + 1e-15*np.eye(n_vars+1)
		L = np.linalg.cholesky(XpI)

	# Repeat rounding for different vectors
	n_rand_vector = 100

	model_vect = np.zeros((n_vars,n_rand_vector))
	obj_vect   = np.zeros(n_rand_vector)

	for kk in range(n_rand_vector):

		# Generate a random cutting plane vector (uniformly 
		# distributed on the unit sphere - normalized vector)
		r = np.random.randn(n_vars+1)
		r = r/np.linalg.norm(r)
		y_soln = np.sign(np.dot(L.T,r))

		# convert solution to original domain and assign to output vector
		model_vect[:,kk] = (y_soln[:n_vars]+1.)/2.
		obj_vect[kk] = np.dot(np.dot(model_vect[:,kk].T,A),model_vect[:,kk]) \
			+ np.dot(b,model_vect[:,kk])

	# Find optimal rounded solution
	opt_idx = np.argmin(obj_vect)
	model = model_vect[:,opt_idx]
	obj   = obj_vect[opt_idx]

	return (model, obj)


# -- END OF FILE --