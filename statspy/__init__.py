"""
StatsPy
=======

Provides
* General classes to define a random variable, a probability density function
  or a parameter
* Tools to perform hypothesis tests or get confidence intervals

Getting started::

    >>> import statspy as sp
    >>> help(sp)

"""
__version__ = "0.1.0a1"

import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError("Python version 2.6 or above is required for StatsPy.")
else:
    pass
del sys

# We first need to detect if we're being called as part of the statspy setup
# procedure itself in a reliable manner.
try:
    __STATSPY_SETUP__
except NameError:
    __STATSPY_SETUP__ = False

if not __STATSPY_SETUP__:
    # Import base classes
    from . import core
    from .core import *

    def test(level=1, verbosity=1):
        from numpy.testing import Tester
        return Tester().test(level, verbosity)
