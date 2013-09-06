#!/usr/bin/env python
"""pllr_interval.py

This example shows how to estimate a confidence interval using an
approximate profile log-likelihood ratio technique.

"""

import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import statspy as sp
import statspy.interval

X = sp.RV("norm(x|mux=20,sigmax=5)")
x = X(size=100)
pdf_fit = sp.PF("pdf_fit=norm(x|mu=1,sigma=1)")
mu = sp.get_obj('mu')
sigma = sp.get_obj('sigma')

# First compute 95% CL bounds
mu.value    = x.mean() - x.std()
sigma.value = x.std() * 0.5
mu.unc      = 0
sigma.unc   = 0
print mu, sigma
params, corr, quantile = statspy.interval.pllr(pdf_fit, x, cl=0.95)
bounds_95cl = []
for par in params:
    bounds_95cl.append([par.value + par.neg_unc, par.value + par.pos_unc])

# Second compute 1-sigma positive/negative uncertainties (68.27% CL)
mu.value    = x.mean() - x.std()
sigma.value = x.std() * 0.5
mu.unc      = 0
sigma.unc   = 0
params, corr, quantile = statspy.interval.pllr(pdf_fit, x)

print 'Confidence intervals summary:'
print 'name = value +pos_unc -neg_unc (parabolic unc)\t[lower, upper] @ 95%CL'
for par,bounds in zip(params,bounds_95cl):
    print par.name,'=',par.value,'+',par.pos_unc,'-',math.fabs(par.neg_unc),
    print '(',par.unc,')\t','[',bounds[0],',',bounds[1],']'
print 'Correlation matrix:\n',corr
