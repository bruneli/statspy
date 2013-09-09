"""module hypotest.py

This module contains functions to perform hypothesis tests.

"""

import logging
import math
import scipy.special
import scipy.stats
import statspy as sp

__all__ =['Result','pvalue_to_Zvalue','Zvalue_to_pvalue','pllr']

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_ch = logging.StreamHandler() # Console handler
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)
logger.debug('logger has been set to DEBUG mode')

class Result(dict):
    """Class to store results from an hypothesis test.

    Among the variables stored in this class, there are:

    * the ``pvalue`` which is defined as Prob(t >= tobs | H0) with t the test
      statistics. If this value is lower than the predefined type I error 
      rate, then the null hypothesis is rejected.
    * the ``Zvalue``, the standard score (or Z-score), is the p-value expressed
      in the number of standard deviations.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

def pvalue_to_Zvalue(pvalue, mode='two-sided'):
    """Convert a p-value to a Z-value.

    Definition::

        math.sqrt(2.) * scipy.special.erfcinv(mode * pvalue)

    mode is equal to 2 for a two-sided Z-value and to 1 for a one-sided
    Z-value.

    Parameters
    ----------
    pvalue : float
        p-value defined as Prob(t >= tobs | H0)
    mode : str
        'two-sided' (default) or 'one-sided'

    Returns
    -------
    Zvalue : float
        Z-value corresponding to the number of standard deviations

    """
    try:
        mode = 2. if mode == 'two-sided' else 1.
        if pvalue < 1.24e-15:
            logger.warning('when pvalue is lower than 1.24e-15, Z-value is set to 8.')
            Zvalue = 8.
        else:
            Zvalue = math.sqrt(2.) * scipy.special.erfcinv(mode * pvalue)
    except:
        raise
    return Zvalue

def Zvalue_to_pvalue(Zvalue, mode='two-sided'):
    """Convert a Z-value to a p-value.

    Definition::

        scipy.special.erfc(Zvalue / math.sqrt(2.)) / mode

    mode is equal to 2 for a two-sided Z-value and to 1 for a one-sided
    Z-value.

    Parameters
    ----------
    Zvalue : float
        Z-value corresponding to the number of standard deviations
    mode : str
        'two-sided' (default) or 'one-sided'

    Returns
    -------
    pvalue : float
        p-value defined as Prob(t >= tobs | H0)

    """
    try:
        mode = 2. if mode == 'two-sided' else 1.
        pvalue = scipy.special.erfc(Zvalue / math.sqrt(2.)) / mode
    except:
        raise
    return pvalue

def pllr(pf, data, **kw):
    """Profile likelihood ratio test.

    For the likelihood ratio test, the likelihood is maximized separately for
    the null and the alternative hypothesis. The word "profile" means that
    in addition, the likelihood is maximized wrt the nuisance parameters.
    The test statistics is then defined as::

        l = L(x|theta_r,\hat{\hat{theta_s}}) / L(x|\hat{theta_r},\hat{theta_s})
        q_obs = -2 * log(l)

    and is distributed asymptotically as a chi2 distribution.
    q_obs can be used to compute a p-value = Pr(q >= q_obs). 

    Parameters
    ----------
    pf : statspy.core.PF
        Probability function used in the computed of the likelihood
    data : ndarray, tuple
        x - variates used in the computation of the likelihood 
    kw : keyword arguments (optional)

    Returns
    -------
    result : statspy.hypotest.Result
        All information about the test is stored in the Result class.

    """
    # Get initial values of parameters
    params = pf.get_list_free_params()
    popt = [None] * len(params)
    for ipar,par in enumerate(params):
        popt[ipar] = par.value

    result = Result()
    result.pf = pf
    result.data = data
    result.pllr = pf.pllr(data)
    result.Zvalue = math.sqrt(result.pllr)
    result.pvalue = scipy.stats.chi2.sf(result.pllr, 1)
    all_discrete = True
    for rv in pf._rvs:
        if rv[3] == sp.RV.DISCRETE: continue
        all_discrete = False
        break
    if all_discrete:
        result.pvalue = result.pvalue + scipy.stats.chi2.pdf(result.pllr, 1)
    logger.debug('PLLR test: pvalue = %f, Zvalue = %f' % (result.pvalue, 
                                                          result.Zvalue))

    # Restore initial value of parameters
    for ipar,par in enumerate(params):
        par.value = popt[ipar]

    return result
