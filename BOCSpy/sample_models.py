# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np

def sample_models(n_models, n_vars):
	# SAMPLE_MODELS: Function samples the binary models to
	# generate observations to train the statistical model

	# Generate matrix of zeros with ones along diagonals
	binary_models = np.zeros((n_models, n_vars))

	# Sample model indices
	model_num = np.random.randint(2**n_vars, size=n_models)

	strformat = '{0:0' + str(n_vars) + 'b}'
	# Construct each binary model vector
	for i in range(n_models):
		model = strformat.format(model_num[i])
		binary_models[i,:] = np.array([int(b) for b in model])

	return binary_models

# -- END OF FILE --