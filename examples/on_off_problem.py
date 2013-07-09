#!/usr/bin/env python

"""on_off_problem.py

This example compares Z values from various hypothesis tests performed
when incorporating a systematic uncertainty into a test of the
background-only hypothesis for a Poisson process.
The on/off problem, the different statistical tests and the input/output values
are described in:
    http://arxiv.org/abs/physics/0702156

"""

import numpy as np
import statspy as sp
import matplotlib.pyplot as plt
from matplotlib import ticker

# Define the different parameters of the problem
# Parameter of interest:
s = sp.Param(name='s',value=0) # Signal expectation in the signal region
# Nuisance parameter:
b = sp.Param(name='b',value=1) # Background expectation in the signal region
# Transfer factor between the control and signal regions (constant)
tau = sp.Param(name='tau',value=5)
# Derived quantities
mu_on  = s + b # Total events expectation in the signal region
mu_off = tau*b # Total events expectation in the control region
mu_on.name  = 'mu_on'  # Parameters must be named to be recalled later via str
mu_off.name = 'mu_off'

# Define the probability mass functions corresponding to n_on and n_off
pmf_on  = sp.PF('pmf_on=poisson(n_on;mu_on)')
pmf_off = sp.PF('pmf_off=poisson(n_off;mu_off)')
likelihood = pmf_on * pmf_off

# 2D-histo of likelihood vs (n_on, n_off)
x = y = np.arange(0, 20)
X, Y = np.meshgrid(x, y) # Build 2D-arrays from 1D-arrays
Z = likelihood(X,Y)
levels = [0.,0.2,0.4,0.6,0.8,1.]
fig = plt.figure()
ax = fig.add_subplot(111)
#cp = plt.pcolor(X, Y, z)
cp = plt.contourf(X, Y, Z, locator=ticker.LogLocator())
plt.xlabel(likelihood._rvs[0])
plt.ylabel(likelihood._rvs[1])
cbar = plt.colorbar(cp)
cbar.ax.set_ylabel('likelihood')
plt.show()
