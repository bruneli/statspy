"""
StatsPy
=======

Provides
  * General classes to define a random variable, a probability density function    or a parameter
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

# Import base classes
import core
from core import *
