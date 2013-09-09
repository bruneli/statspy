.. -*- mode: rst -*-

=======
StatsPy
=======

StatsPy is a Python module for statistics built on top of `NumPy/SciPy <http://docs.scipy.org/doc/>`_. It contains a list of classes and tools intended to ease the design of complex distributions which can be later used to perform statistical tests or extract confidence intervals.

Links
=====

- Source code repository: https://github.com/bruneli/statspy
- User's and Reference Guides: http://bruneli.github.io/statspy/
- Download releases: https://sourceforge.net/projects/statspy/files

Install
=======

After downloading/checking out files, to install you should run the ``setup.py`` script::

    python setup.py install

If you are not familiar with Distutils, have a look to the `official documentation <http://docs.python.org/2/install/>`_. 

Development
===========

GIT
---

Latest sources can be checked out via:

    git clone git://github.com:bruneli/statspy.git

or (if you have write privileges):

    git clone git@github.com:bruneli/statspy.git


TODO
----

   * *PF*:
       * Multivariate PFs
       * Improve bounds for rv operations
   * *Param*:
       * For bayesian stats, add prior (posterior) 
       * Add missing math functions (fabs, trigonometric, hyperbolic)
   * *RV*:
       * Operations: multiplication, division, power, exp, log
   * *Hypothesis tests*:
       * Improve handling of hypothesis
       * Bayesian/Frequentist Hybrid
       * Bayesian tests
   * *Confidence intervals*:
       * Neyman constructions
       * Bayesian credibility intervals
