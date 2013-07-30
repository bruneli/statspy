#!/usr/bin/env python

"""on_off_problem.py

This example compares Z values from various hypothesis tests performed
when incorporating a systematic uncertainty into a test of the
background-only hypothesis for a Poisson process.
The on/off problem, the different statistical tests and the input/output values
are described in:
    http://arxiv.org/abs/physics/0702156

"""

import math
import numpy as np
import statspy as sp
import matplotlib.pyplot as plt
from matplotlib import ticker

# Define the different parameters of the problem
# Parameter of interest:
mu = sp.Param(name='mu', value=0, poi=True)  # Signal strength
s  = sp.Param(name='s', value=3, const=True) # Expected number of signal evts
# Nuisance parameter:
b = sp.Param(name='b', value=1) # Expected number of bkg evts in signal region
# Transfer factor between the control and signal regions (constant)
tau = sp.Param(name='tau', value=5, const=True)
# Derived quantities
mu_on  = mu * s + b # Total events expectation in the signal region
mu_off = tau*b      # Total events expectation in the control region
mu_on.name  = 'mu_on'  # Parameters must be named to be recalled later via str
mu_off.name = 'mu_off'

# Define the probability mass functions corresponding to n_on and n_off
pmf_on  = sp.PF('pmf_on=poisson(n_on;mu_on)')
pmf_off = sp.PF('pmf_off=poisson(n_off;mu_off)')
likelihood = pmf_on * pmf_off
data = (4, 5) #(pmf_on.rvs(size=1), pmf_off.rvs(size=1))
print 'data',data
#likelihood.maxlikelihood_fit(data)
pllr = likelihood.pllr(data)
print 'pllr',pllr,math.sqrt(pllr)

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
