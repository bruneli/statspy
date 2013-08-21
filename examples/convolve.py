#!/usr/bin/env python

"""convolve.py

This example shows how to convolve two PFs.

"""

import matplotlib.pyplot as plt
import numpy as np
import statspy as sp

pdf1 = sp.PF("pdf1=expon(x;offset1=0,tau1=10)")
pdf2 = sp.PF("pdf2=expon(x;offset2=0,tau2=20)")
#pdf3 = sp.PF("pdf2=expon(x;offset3=0,tau3=10)")
#pdf1 = sp.PF("pdf1=norm(x;mu1=40,sigma1=3)")
#pdf2 = sp.PF("pdf2=norm(x;mu2=-10,sigma2=4)")
#pdf1 = sp.PF("pdf1=poisson(n;mu1=3)")
#pdf2 = sp.PF("pdf2=poisson(n;mu2=15)")
pdf3 = pdf1.rvsub(pdf2, mode='fft')
pdf4 = pdf1.rvsub(pdf2, mode='num')
pdf5 = pdf1.rvsub(pdf2, mode='rvs')
pdf6 = sp.PF("pdf6=norm(x;mu6=20,sigma6=5)")
#pdf6 = sp.PF("pdf6=poisson(n;mu6=15)")

fig = plt.figure()
fig.patch.set_color('w')
ax = fig.add_subplot(111)

#x = np.asarray(range(40))
x = np.linspace(-30., 30., 200)
#x = np.linspace(pdf3._cache.x[0], pdf3._cache.x[-1], 200)
#x = np.linspace(pdf3._cache.n[0], pdf3._cache.n[-1], int(pdf3._cache.n.shape[0]))
ax.plot(x, pdf1(x), label='pdf1')
ax.plot(x, pdf2(x), label='pdf2')
ax.plot(x, pdf3(x), label='fft')
ax.plot(x, pdf4(x), label='num')
ax.plot(x, pdf5(x), label='rvs')
#ax.plot(x, pdf6(x), label='analytic')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

#print 'loc',pdf3.loc.value,pdf4.loc.value
plt.show()
