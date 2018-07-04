# Author: Ricardo Baptista and Matthias Poloczek
# Date:   June 2018
#
# See LICENSE.md for copyright information
#

import numpy as np

def quad_mat(n_vars, alpha):
 
	# evaluate decay function
	i = np.linspace(1,n_vars,n_vars)
	j = np.linspace(1,n_vars,n_vars)
	
	K = lambda s,t: np.exp(-1*(s-t)**2/alpha)
	decay = K(i[:,None], j[None,:])

	# Generate random quadratic model
	# and apply exponential decay to Q
	Q  = np.random.randn(n_vars, n_vars)
	Qa = Q*decay
	
	return Qa

# -- END OF FILE --