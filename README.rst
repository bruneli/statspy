.. -*- mode: rst -*-

StatsPy
=======

StatsPy is a Python module for statistics built on top of NumPy/SciPy. It contains a list of classes and tools intended to ease the design of complex distributions which can be later used to perform statistical tests or extract confidence intervals.

Links
=====

- Source code repository: https://github.com/bruneli/statspy
- User's Guide: https://github.com/bruneli/statspy/blob/master/doc/MANUAL.md
- Reference Guide: <tobedone>

Install
=======

After downloading/checking out files, to install you should run the ``setup.py`` script:

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
      * Max likelihood fit
   * *Param*:
      * Uncertainty propagation
   * *RV*:
      * Multiplication/Division/Power
   * *Hypothesis tests*:
      * PLLR test
   * *Confidence intervals*:
      * PLLR-based intervals (MINOS)
      * Neyman constructions
      * Bayesian credibility intervals
      * Bayesian/Frequentist Hybrid
