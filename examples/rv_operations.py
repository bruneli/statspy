#!/usr/bin/env python

"""rv_operations.py

This example shows how to add, subtract, multiply or divide random variables.

"""

import matplotlib.pyplot as plt
import numpy as np
import statspy as sp

# Input Normal PDFs
X = sp.RV("norm(x;mux=15,sigmax=3)")
Y = sp.RV("norm(x;muy=10,sigmay=4)")

fig = plt.figure()
fig.patch.set_color('w')

# Addition
Z1 = X + Y
ax_add = fig.add_subplot(212)
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
ax_sub = fig.add_subplot(211)
mean_Z2 = Z2.pf.mean()
rms_Z2 = Z2.pf.std()
z2 = np.linspace(mean_Z2 - 5 * rms_Z2, mean_Z2 + 5 * rms_Z2, 200)
ax_sub.plot(z2, X.pf(z2), label='X')
ax_sub.plot(z2, Y.pf(z2), label='Y')
ax_sub.plot(z2, Z2.pf(z2), label='X-Y')
handles, labels = ax_sub.get_legend_handles_labels()
ax_sub.legend(handles[::-1], labels[::-1])

plt.show()
