#!/usr/bin/env python

"""curve_fitting.py

This example shows how to fit a PDF to data.

"""

import numpy as np
import statspy as sp
import matplotlib.pyplot as plt

# Define the true PDF
pdf_true = sp.PF("pdf_true=norm(x;mu_true=10,sigma_true=2)")
# Sample data from the true PDF
data = pdf_true.rvs(size=1000)

fig = plt.figure()
ax = fig.add_subplot(111)

# Build histogram of the data
ydata, bins, patches = ax.hist(data, 10, facecolor='green', alpha=0.75)
xdata = 0.5*(bins[1:]+bins[:-1])
print xdata, ydata

# Least square fit to the data

plt.show()
