""" Test functions for the statspy package

"""

import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal

import statspy as spy

class TestRandomVariable(unittest.TestCase):
    """ Test functions for the statspy.core.RV class """

    def test_param_value_initialization(self):
        """Test that two different ways of setting parameter values
           give the same answer.

        """
        a = np.array([2.,6.,8.,10.,12.,14.,18.])
        x = spy.RV("norm(x;mu_x=10,sigma_x=2)")
        y = spy.RV("norm(y|mu_y,sigma_y)",mu_y=10.,sigma_y=2.)
        assert_array_almost_equal(x.pdf(a),y.pdf(a))

if __name__ == "__main__":
    unittest.main()
