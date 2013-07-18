#!/usr/bin/env python

"""curve_fitting.py

This example shows how to fit a PDF to data.

"""

import matplotlib.pyplot as plt
import numpy as np
import statspy as sp
import scipy.stats

# Define the true PDF
pdf_true_sig = sp.PF("pdf_true_sig=norm(x;mu_true=125,sigma_true=10)")
pdf_true_bkg = sp.PF("pdf_true_bkg=expon(x;offset=0,lambda_true=10)")
pdf_true = pdf_true_sig + pdf_true_bkg
pdf_true_bkg.norm.value = 0.9
# Sample data from the true PDF
nexp = 10000 # number of expected events
nobs = scipy.stats.poisson.rvs(nexp, size=1)
data = pdf_true.rvs(size=nobs)
# Define the PF to fit
pdf_fit = sp.PF("pdf_fit=norm(x;mu=1,sigma=1)")

fig = plt.figure()
fig.patch.set_color('w')
ax = fig.add_subplot(111)

# Build histogram of the data
ydata, bins, patches = ax.hist(data, 50, facecolor='green', alpha=0.75,
                               label='Data')
xdata = 0.5*(bins[1:]+bins[:-1]) # Bin centers
dx = bins[1:] - bins[:-1]        # Bin widths

# Least square fit to the data
#params, pcov, chi2min, pvalue = pdf_fit.leastsq_fit(xdata, ydata, dx=dx)
#yfit  = pdf_fit(xdata) * dx
#eyfit = pdf_fit.dF(xdata) * dx # Get error bars on the fitted PF
#ax.plot(xdata, yfit, 'r--', linewidth=2, label='Fitted PF')
#ax.fill_between(xdata, yfit-eyfit, yfit+eyfit, facecolor='y')
ax.plot(xdata, pdf_true(xdata) * dx * nexp, 'b:', linewidth=2, label='True PF')

# Plot
ax.set_xlabel('x')
ax.set_ylabel('Evts / %3.2f' % dx[0])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
#fittxt = "Fit results:\n"
#fittxt += "$\Delta\chi^{2}/ndf = %3.2f$\n" % (chi2min/(len(ydata)-len(params)))
#fittxt += "$p-value = %3.2f$\n" % pvalue
#for par in params:
#    fittxt += "$%s = %3.2f \\pm %3.2f$\n" % (par.name, par.value, par.unc)
#ax.text(0.05, 0.95, fittxt, transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox={"boxstyle":"square","fc":"w"})

plt.show()
