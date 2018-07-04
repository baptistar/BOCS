#
# Bayesian Optimization of Combinatorial Structures
#
# Copyright (C) 2018 R. Baptista & M. Poloczek
# 
# BOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with BOCS.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2018 MIT & University of Arizona
# Authors: Ricardo Baptista & Matthias Poloczek
# E-mails: rsb@mit.edu & poloczek@email.arizona.edu
#

import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from quad_mat import quad_mat
from sample_models import sample_models

# Save inputs in dictionary
inputs = {}
inputs['n_vars']     = 10
inputs['evalBudget'] = 50
inputs['n_init']     = 10
inputs['lambda']     = 1e-4

# Save objective function and regularization term
Q = quad_mat(inputs['n_vars'], 10)
inputs['model']    = lambda x: (x.dot(Q)*x).sum(axis=1) # compute x^TQx row-wise
inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)

# Generate initial samples for statistical models
inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
inputs['y_vals']   = inputs['model'](inputs['x_vals'])

# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size)
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)

# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
for i in range(n_models):
	model = str_format.format(i)
	x_vals[i,:] = np.array([int(b) for b in model])
f_vals = inputs['model'](x_vals) + inputs['penalty'](x_vals)
opt_f  = np.min(f_vals)

# Plot results
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(iter_t, np.abs(BOCS_SA_opt - opt_f), color='r', label='BOCS-SA')
ax.plot(iter_t, np.abs(BOCS_SDP_opt - opt_f), color='b', label='BOCS-SDP')
ax.set_yscale('log')
ax.set_xlabel('$t$')
ax.set_ylabel('Best $f(x)$')
ax.legend()
fig.savefig('BOCS_simpleregret.pdf')
plt.close(fig)

# -- END OF FILE --