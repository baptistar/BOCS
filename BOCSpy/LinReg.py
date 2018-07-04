# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np
from itertools import combinations
from bhs import bhs

class LinReg:

	def __init__(self, nVars, order):
		self.nVars = nVars
		self.order = order

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def setupData(self):

		# limit data to unique points
		X, x_idx = np.unique(self.xTrain, axis=0, return_index=True)
		y = self.yTrain[x_idx]

		# set upper threshold
		infT = 1e6

		# separate samples based on Inf output
		y_Infidx  = np.where(np.abs(y) > infT)[0]
		y_nInfidx = np.setdiff1d(np.arange(len(y)), y_Infidx)

		# save samples in two sets of variables
		self.xInf = X[y_Infidx,:]
		self.yInf = y[y_Infidx]

		self.xTrain = X[y_nInfidx,:]
		self.yTrain = y[y_nInfidx]

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def train(self, inputs):

		# Set nGibbs (Gibbs iterations to run)
		nGibbs = int(1e3)

		# set data
		self.xTrain = inputs['x_vals']
		self.yTrain = inputs['y_vals']

		# setup data for training
		self.setupData()

		# create matrix with all covariates based on order
		self.xTrain = self.order_effects(self.xTrain, self.order)
		(nSamps, nCoeffs) = self.xTrain.shape

		# check if x_train contains columns with zeros or duplicates
		# and find the corresponding indices
		check_zero = np.all(self.xTrain == np.zeros((nSamps,1)), axis=0)
		idx_zero   = np.where(check_zero == True)[0]
		idx_nnzero = np.where(check_zero == False)[0]

		# remove columns of zeros in self.xTrain
		if np.any(check_zero):
			self.xTrain = self.xTrain[:,idx_nnzero]

		# run Gibbs sampler for nGibbs steps
		attempt = 1
		while(attempt):

			# re-run if there is an error during sampling
			try:
				alphaGibbs,a0,_,_,_ = bhs(self.xTrain,self.yTrain,nGibbs,0,1)
			except:
				print('error during Gibbs sampling. Trying again.')
				continue

			# run until alpha matrix does not contain any NaNs
			if not np.isnan(alphaGibbs).any():
				attempt = 0
			
		# append zeros back - note alpha(1,:) is linear intercept
		alpha_pad = np.zeros(nCoeffs)
		alpha_pad[idx_nnzero] = alphaGibbs[:,-1]
		self.alpha = np.append(a0, alpha_pad)

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def surrogate_model(self, x, alpha):
		# SURROGATE_MODEL: Function evaluates the linear model
		# Assumption: input x only contains one row

		# generate x_all (all basis vectors) based on model order
		x_all = np.append(1, self.order_effects(x, self.order))

		# check if x maps to an Inf output (if so, barrier=Inf)
		barrier = 0.
		if self.xInf.shape[0] != 0:
			if np.equal(x, self.xInf).all(axis=1).any():
				barrier = np.inf

		# compute and return objective with barrier
		out = np.dot(x_all,alpha) + barrier
		
		return out

	# ---------------------------------------------------------
	# ---------------------------------------------------------

	def order_effects(self, x_vals, ord_t):
		# order_effects: Function computes data matrix for all coupling
		# orders to be added into linear regression model.

		# Find number of variables
		n_samp, n_vars = x_vals.shape

		# Generate matrix to store results
		x_allpairs = x_vals

		for ord_i in range(2,ord_t+1):

			# generate all combinations of indices (without diagonals)
			offdProd = np.array(list(combinations(np.arange(n_vars),ord_i)))

			# generate products of input variables
			x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
			for j in range(ord_i):
				x_comb[:,:,j] = x_vals[:,offdProd[:,j]]
			x_allpairs = np.append(x_allpairs, np.prod(x_comb,axis=2),axis=1)

		return x_allpairs


# -- END OF FILE --