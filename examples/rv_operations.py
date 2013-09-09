#!/usr/bin/env python

"""rv_operations.py

This example shows how to rescale, add, subtract, multiply or divide random 
variables.

"""

import matplotlib.pyplot as plt
import numpy as np
import statspy as sp

# Input Normal PDFs
X = sp.RV("norm(x;mux=15,sigmax=3)")
Y = sp.RV("norm(y;muy=10,sigmay=4)")

fig = plt.figure()
fig.patch.set_color('w')

# Addition
Z1 = X + Y
ax_add = fig.add_subplot(321)
mean_Z1 = Z1.pf.mean()
rms_Z1 = Z1.pf.std()
z1 = np.linspace(mean_Z1 - 5 * rms_Z1, mean_Z1 + 5 * rms_Z1, 200)
ax_add.plot(z1, X.pf(z1), label='X')
ax_add.plot(z1, Y.pf(z1), label='Y')
ax_add.plot(z1, Z1.pf(z1), label='X+Y')
handles, labels = ax_add.get_legend_handles_labels()
ax_add.legend(handles[::-1], labels[::-1])

# Subtraction
Z2 = X - Y
ax_sub = fig.add_subplot(322)
mean_Z2 = Z2.pf.mean()
rms_Z2 = Z2.pf.std()
z2 = np.linspace(mean_Z2 - 5 * rms_Z2, mean_Z2 + 5 * rms_Z2, 200)
ax_sub.plot(z2, X.pf(z2), label='X')
ax_sub.plot(z2, Y.pf(z2), label='Y')
ax_sub.plot(z2, Z2.pf(z2), label='X-Y')
handles, labels = ax_sub.get_legend_handles_labels()
ax_sub.legend(handles[::-1], labels[::-1], loc='upper left')

# Multiplication
X3 = sp.RV("norm(x3;mux3=2.0,sigmax3=0.3)")
Y3 = sp.RV("norm(y3;muy3=1.0,sigmay3=0.2)")
Z3 = X3 * Y3
ax_mul = fig.add_subplot(323)
mean_Z3 = Z3.pf.mean()
rms_Z3 = Z3.pf.std()
z3 = np.linspace(mean_Z3 - 5 * rms_Z3, mean_Z3 + 5 * rms_Z3, 200)
ax_mul.plot(z3, X3.pf(z3), label='X')
ax_mul.plot(z3, Y3.pf(z3), label='Y')
ax_mul.plot(z3, Z3.pf(z3), label='X*Y')
handles, labels = ax_mul.get_legend_handles_labels()
ax_mul.legend(handles[::-1], labels[::-1])

# Division
X4 = sp.RV("norm(x4;mux4=1.,sigmax4=0.05)")
Y4 = sp.RV("norm(y4;muy4=1.,sigmay4=0.05)")
Z4 = X4 / Y4
ax_div = fig.add_subplot(324)
mean_Z4 = Z4.pf.mean()
rms_Z4 = Z4.pf.std()
z4 = np.linspace(mean_Z4 - 5 * rms_Z4, mean_Z4 + 5 * rms_Z4, 200)
ax_div.plot(z4, X4.pf(z4), label='X')
ax_div.plot(z4, Y4.pf(z4), label='Y')
ax_div.plot(z4, Z4.pf(z4), label='X/Y')
handles, labels = ax_div.get_legend_handles_labels()
ax_div.legend(handles[::-1], labels[::-1])

# Rescaling
mux = sp.get_obj('mux')
sigmax = sp.get_obj('sigmax')
Z5 = (X - mux) / sigmax
ax_res = fig.add_subplot(325)
z5 = np.linspace(mux.value - 7 * sigmax.value, mux.value + 3 * sigmax.value, 200)
ax_res.plot(z5, X.pf(z5), label='X')
ax_res.plot(z5, Z5.pf(z5), label='(X-$\mu$)/$\sigma$')
handles, labels = ax_res.get_legend_handles_labels()
ax_res.legend(handles[::-1], labels[::-1])

plt.show()
fig.savefig('rv_operations.png', dpi=fig.dpi)
