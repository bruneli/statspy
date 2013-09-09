#!/usr/bin/env python
"""on_off_problem.py

This example compares Z values from various hypothesis tests performed
when incorporating a systematic uncertainty into a test of the
background-only hypothesis for a Poisson process.
The on/off problem, the different statistical tests and the input/output values
are described in http://arxiv.org/abs/physics/0702156

"""

import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import statspy as sp
import statspy.hypotest

# Experimental data stored as (n_on, n_off, tau)
exps = [[4, 5, 5.],
        [6, 18.78, 14.44],
        [9, 17.83, 4.69],
        [17, 40.11, 10.56],
        [50, 55, 2.],
        [67, 15, 0.5],
        [200, 10, 0.1],
        [523, 2327, 5.99],
        [498426, 493434, 1.],
        #[211949, 23650096, 11.21], # numerical instabilities with this exp
        ]

# Define the different parameters of the problem
# Parameter of interest:
mu = sp.Param(name='mu', value=0., poi=True)  # Signal strength
s  = sp.Param(name='s', value=3., const=True) # Expected number of signal evts
# Nuisance parameter:
b = sp.Param('b=1.') # Expected number of bkg evts in the signal region
# Transfer factor between the control and signal regions (constant)
tau = sp.Param(name='tau', value=5., const=True)
# With normal bkg shape, standard deviation on b
sigmab = sp.Param('sigmab=0.447', const=True)
# Derived quantities
mu_on  = mu * s + b # Total events expectation in the signal region
mu_off = tau*b      # Total events expectation in the control region
mu_on.name  = 'mu_on'  # Parameters must be named to be recalled later via str
mu_off.name = 'mu_off'
rho = 1./(1. + tau)
rho.name = 'rho'

# Define the probability mass function corresponding to n_on
pmf_on  = sp.PF('pmf_on=poisson(n_on;mu_on)')
# Define the probability mass function of n_off as a Poisson
pmf_off = sp.PF('pmf_off=poisson(n_off;mu_off)')
likelihood_P = pmf_on * pmf_off
# Approximate the distribution of b as a Gaussian of width sigmab
pdf_off = sp.PF('pdf_off=norm(x;b,sigmab)')
likelihood_G = pmf_on * pdf_off
# The joint distribution likelihood_P can be rewritten as the product of a
# poisson distribution on n_tot = n_on + n_off and a binomial distribution
# with parameter rho. The binomial part is used to compute a p-value in a
# Frequentist approach.
n_tot = sp.Param('n_tot=9.', const=True)
pmf_ratio = sp.PF('pmf_ratio=binom(n_on;n_tot,rho)')
    
# Loop over experiments
for exp in exps:
    # Initialize RAW parameters for that experiment
    (n_on, n_off, tau0) = exp
    mu.value = 0. # Test a background only null hypothesis
    tau.value = tau0
    b.value = n_off / tau0
    s.value = n_on - b.value
    sigmab.value = math.sqrt(b.value) / math.sqrt(tau0)
    
    # Compute Z-value in case of no uncertainty on b
    Z_P = statspy.hypotest.pvalue_to_Zvalue(pmf_on.pvalue(n_on))
    exp.append(Z_P)

    #
    # Frequentist solution
    #
    n_tot.value = n_on + round(n_off)
    Z_Bi = statspy.hypotest.pvalue_to_Zvalue(pmf_ratio.pvalue(n_on))
    exp.append(Z_Bi)

    #
    # Profile likelihood methods
    #
    data_P = (n_on, round(n_off))
    res_pllr_P = statspy.hypotest.pllr(likelihood_P, data_P)
    exp.append(res_pllr_P.Zvalue)
    data_G = (n_on, n_off/tau0)
    res_pllr_G = statspy.hypotest.pllr(likelihood_G, data_G)
    exp.append(res_pllr_G.Zvalue)

for idx,var in enumerate(['n_on','n_off','tau','Z_P','Z_Bi','Z_PLP','Z_PLG']):
    the_str = var
    for exp in exps:
        the_str = the_str + '\t%5.2f' % exp[idx]
    print the_str
