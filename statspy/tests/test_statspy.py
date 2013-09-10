""" Test functions for the statspy package

"""

import math
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
import statspy as sp
import unittest

class TestParameter(unittest.TestCase):
    """ Test functions for the statspy.core.Param class """

    def test_operations(self):
        """Test that parameter operations are fine."""   
        x = sp.Param("x = 2. +- 1.")
        y = sp.Param("y = 4. +- 1.")
        z = np.zeros(8)
        z[0] = (x + y).value
        z[1] = (x - y).value
        z[2] = (x * y).value
        z[3] = (x / y).value
        z[4] = (x**2).value
        z[5] = (sp.sqrt(x)).value
        z[6] = (sp.exp(x)).value
        z[7] = (sp.log(x)).value
        result = np.asarray([2. + 4., # x+y
                             2. - 4., # x-y
                             2. * 4., # x*y
                             2. / 4., # x/y
                             2.**2,   # x**2
                             math.sqrt(2.), # sqrt(x)
                             math.exp(2.),  # exp(x)
                             math.log(2.),  # log(x)
                             ])
        assert_array_equal(z, result)

    def test_uncertainty_propagation(self):
        """Test that uncertainty propagation formula are fine."""
        x = sp.Param("x = 2. +- 1.")
        y = sp.Param("y = 4. +- 1.")
        z = np.zeros(5)
        z[0] = (x + y).unc
        z[1] = (x - y).unc
        z[2] = (x * y).unc
        z[3] = (x / y).unc
        z[4] = (x**2).unc
        result = np.asarray([math.sqrt(2.), # x+y
                             math.sqrt(2.), # x-y
                             math.sqrt(4.*4. + 2.*2.), # x*y
                             math.sqrt(1./4./4. + 2.*2./math.pow(4.,4)), # x/y
                             2.*2.]) # x**2
        assert_array_equal(z, result)

class TestRandomVariable(unittest.TestCase):
    """ Test functions for the statspy.core.RV class """

    def test_param_value_initialization(self):
        """Test that two different ways of setting parameter values
           give the same answer.

        """
        a = np.linspace(2., 18., 9)
        X1 = sp.RV("norm(x1;mu_x1=10,sigma_x1=2)")
        Y1 = sp.RV("norm(y1|mu_y1,sigma_y1)",mu_y1=10.,sigma_y1=2.)
        assert_array_equal(X1.pf(a), Y1.pf(a))

    def test_rv_addition(self):
        """Test addition of two normal distributions gives the
           correct resulting normal distribution.

        """
        X2 = sp.RV("norm(x2;mu_x2=10,sigma_x2=3)")
        Y2 = sp.RV("norm(y2;mu_y2=20,sigma_y2=4)")
        Z1 = sp.RV("norm(z1|mu_z1=30,sigma_z1=5)")
        Z2 = X2 + Y2 
        z = np.linspace(15.,45.,200)
        # precision abs(Z1.pf(z)-Z2.pf(z)) < 0.5 * 10**(-decimal)
        assert_array_almost_equal(Z1.pf(z), Z2.pf(z), decimal=3)

    def test_rv_subtraction(self):
        """Test subtraction of two normal distributions gives the
           correct resulting normal distribution.

        """
        X3 = sp.RV("norm(x3;mu_x3=10,sigma_x3=3)")
        Y3 = sp.RV("norm(y3;mu_y3=20,sigma_y3=4)")
        Z3 = sp.RV("norm(z3|mu_z1=30,sigma_z3=5)")
        Z4 = X3 + Y3 
        z = np.linspace(15.,45.,200)
        # precision abs(Z4.pf(z)-Z3.pf(z)) < 0.5 * 10**(-decimal)
        assert_array_almost_equal(Z3.pf(z), Z4.pf(z), decimal=3)

    def test_rv_rescaling(self):
        """Test rescaling of a normal distributions gives the
           correct resulting normal distribution.

        """
        X5 = sp.RV("norm(x5;mu_x5=10,sigma_x5=3)")
        Y5 = (X5 - 10.) / 3.
        Z5 = sp.RV("norm(z5|mu_z5=0,sigma_z5=1)")
        z = np.linspace(-2.,2.,200)
        # precision abs(Y5.pf(z)-Z5.pf(z)) < 0.5 * 10**(-decimal)
        assert_array_almost_equal(Y5.pf(z), Z5.pf(z), decimal=3)

if __name__ == "__main__":
    unittest.main()
