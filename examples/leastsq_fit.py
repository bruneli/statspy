#!/usr/bin/env python
"""leastsq_fit.py

This example shows how to fit a PDF to data via the method of 
least-squares.

"""

import matplotlib.pyplot as plt
import numpy as np
import statspy as sp
import scipy.stats

# Define the true PDF (gaussian signal on top of an exponentially falling bkg)
pdf_true_sig = sp.PF("pdf_true_sig=norm(x;mu_true=125,sigma_true=10)")
pdf_true_bkg = sp.PF("pdf_true_bkg=expon(x;offset_true=50,lambda_true=20)")
pdf_true = 0.95 * pdf_true_bkg + 0.05 * pdf_true_sig
# Sample data from the true PDF
nexp = 2000 # number of expected events
nobs = scipy.stats.poisson.rvs(nexp, size=1)[0]
data = pdf_true.rvs(size=nobs)

fig = plt.figure()
fig.patch.set_color('w')
ax = fig.add_subplot(111)

# Build histogram of the data
ydata, bins, patches = ax.hist(data, 30, range=[50, 200], log=True,
                               facecolor='green', alpha=0.75, label='Data')
xdata = 0.5*(bins[1:]+bins[:-1]) # Bin centers
dx = bins[1:] - bins[:-1]        # Bin widths

# Define the background and signal PFs
pdf_fit_bkg = sp.PF("pdf_fit_bkg=expon(x;offset=50,lambda=10)")
sp.get_obj("lambda").label = "\\lambda"
offset = sp.get_obj('offset')
offset.const = True # Fix parameter value
#pdf_fit_bkg.norm.value = nobs
#sidebands = np.logical_or((xdata < 100), (xdata > 150))
#pdf_fit_bkg.leastsq_fit(xdata, ydata, dx=dx, cond=sidebands)
pdf_fit_sig = sp.PF("pdf_fit_sig=norm(x;mu=120,sigma=20)")
sp.get_obj("mu").label    = "\\mu"
sp.get_obj("sigma").label = "\\sigma"
pdf_fit = pdf_fit_bkg + pdf_fit_sig
pdf_fit.name = 'pdf_fit'
pdf_fit.norm.const = False # Fit total rate to data
pdf_fit.norm.label = 'Norm'
pdf_fit.norm.value = nobs
pdf_fit_sig.norm.label = 'frac(sig)'
# Least square fit to the data (whole data range)
params, pcov, chi2min, pvalue = pdf_fit.leastsq_fit(xdata, ydata, dx=dx)
yfit  = pdf_fit(xdata) * dx
eyfit = pdf_fit.dF(xdata) * dx # Get error bars on the fitted PF
ax.plot(xdata, yfit, 'r--', linewidth=2, label='Fitted PF')
ax.fill_between(xdata, yfit-eyfit, yfit+eyfit, facecolor='y')
ax.plot(xdata, pdf_true(xdata) * dx * nexp, 'b:', linewidth=2, label='True PF')

# Plot
ax.set_xlabel('x')
ax.set_ylabel('Evts / %3.2f' % dx[0])
ax.set_xlim(50, 200)
ax.set_ylim(0.1, nexp)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
fittxt = "Fit results:\n"
fittxt += "$\Delta\chi^{2}/ndf = %3.2f$\n" % (chi2min/(len(ydata)-len(params)))
fittxt += "$p-value = %4.3f$\n" % pvalue
for par in params:
    fittxt += "$%s = %3.2f \\pm %3.2f$\n" % (par.label, par.value, par.unc)
ax.text(0.05, 0.05, fittxt, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox={"boxstyle":"square","fc":"w"})

plt.show()
fig.savefig('leastsq_fit.png', dpi=fig.dpi)
