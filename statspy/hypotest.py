"""module hypotest.py

This module contains functions to perform hypothesis tests.

"""

import logging
import math
import numpy as np
import scipy.special
import scipy.stats
import statspy as sp

__all__ =['Result','pvalue_to_Zvalue','Zvalue_to_pvalue','pllr','hybrid']

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_ch = logging.StreamHandler() # Console handler
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)
logger.debug('logger has been set to DEBUG mode')

class Result(dict):
    """Class to store results from an hypothesis test.

    Among the variables stored in this class, there are:

    * the ``pvalue`` which is defined as Prob(t >= t_obs | H0) with t the test
      statistics. If this value is lower than the predefined type I error 
      rate, then the null hypothesis is rejected.
    * the ``Zvalue``, the standard score (or Z-score), is the p-value expressed
      in the number of standard deviations.
    * the observed value of the test statistics ``t_obs`` given data.
    * the probability distribution of the test statistics ``t_pf`` which is 
      used to compute the pvalue and Zvalue.

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
        p-value defined as Prob(t >= t_obs | H0)
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
        p-value defined as Prob(t >= t_obs | H0)

    """
    try:
        mode = 2. if mode == 'two-sided' else 1.
        pvalue = scipy.special.erfc(Zvalue / math.sqrt(2.)) / mode
    except:
        raise
    return pvalue

def pllr(pf, data, **kw):
    """Profile Log-Likelihood Ratio test.

    This test relies on the log-likelihood ratio statistics defined as::

        l = max{L(x|theta) : theta in Theta0} / max{L(x|theta) : theta in Theta}
        t_obs = -2 * log(l)

    with 

    * Theta the ensemble of theta values satisfied both by the null H0 and
      the alternate H1 hypothesis,
    * Theta0 a subset of Theta specified by H0,
    * L the Likelihood function (computed from data).

    Depending on the number of nuisance parameters theta_s and the nature
    of the null H0 and alternative H1 hypothesis, l simplifies to:

    * A simple likelihood ratio when both H0 and H1 are simple::

        l = L(x|theta_r0) / L(x|theta_r1)

      following the Neyman-Pearson Lemma.
    * A likelihood ratio with fixed values at the numerator::

        l = L(x|theta_r0,\hat{\hat{theta_s}}) / L(x|theta_r1,\hat{theta_s})

      when the null hypothesis is taking single values.

    The word "profile" means that the likelihood is maximized wrt the nuisance
    parameters when present.
    Asymptotically, according to the Wilks' theorem, generally the test 
    statistics t is distributed as a chi2 distribution from which a p-value
    Pr(t >= t_obs) can be computed.
    When the chi2 approximation is not met, it is possible to artificially
    generate the distribution of t via pseudo experiments.

    Parameters
    ----------
    pf : statspy.core.PF
        Probability function used in the computed of the likelihood
    data : ndarray, tuple
        x - variates used in the computation of the likelihood 
    kw : keyword arguments (optional)

        mode : str
            can be 'asymptotic' (default) or 'toys' depending whether
            an asymptotic chi2 distribution or a PF generated from toy
            MC is used to compute t_pf.
        h0 : dict
            parameter(s) value(s)/range(s) for the null hypothesis
        h1 : dict
            parameter(s) value(s)/range(s) for the alternate hypothesis
        ntoys : int
            number of pseudo-experiments to generate.

    Returns
    -------
    result : statspy.hypotest.Result
        All information about the test is stored in the Result class.

    """
    # Store the initial values of the parameters
    params = pf.get_list_free_params()
    popt = [None] * len(params)
    for ipar,par in enumerate(params):
        popt[ipar] = par.value

    result = Result()

    # Store inputs
    result.pf = pf
    result.data = data
    result.mode = kw.get('mode','asymptotic')

    # Probability function of the test statistics
    if result.mode == 'asymptotic':
        df = [0, 0]
        for i,hypo in enumerate(['h0','h1']):
            if hypo in kw:
                for key in kw[hypo]:
                    if type(kw[hypo][key]) == list: continue
                    df[i] += 1
            elif hypo == 'h0':
                for par in params:
                    if par.poi: df[i] += 1
        result.t_pf = sp.PF("chi2(t|df)", df=df[0]-df[1])
    else: # Generate a toy MC sample (very slow...)
        result.ntoys = kw.get('ntoys', 10000)
        logger.info('Number of toys = %d', result.ntoys)
        if isinstance(data, np.ndarray):
            nevts = data.shape[0]
        else:
            nevts = 1
        t_exp = np.empty(result.ntoys)
        def get_t_exp():
            for iexp in range(result.ntoys):
                if result.ntoys > 20 and iexp % (result.ntoys / 20) == 0:
                    logger.info('Pseudo exp %d / %d', iexp, result.ntoys)
                for ipar,par in enumerate(params):
                    par.value = popt[ipar]
                pseudo_exp = pf.rvs(size=nevts)
                yield pf.pllr(pseudo_exp, **kw)
        for i, el in enumerate(get_t_exp()): t_exp[i] = el
        result.t_pf = sp.PF(hist=np.histogram(t_exp, bins=200))

    # Compute the observed value of the test statistics
    result.t_obs = pf.pllr(data, **kw)

    # Extract the pvalue and the Zvalue
    result.Zvalue = math.sqrt(result.t_obs)
    result.pvalue = result.t_pf.pvalue(result.t_obs)
    logger.info('PLLR test: pvalue = %f, Zvalue = %f' % (result.pvalue, 
                                                         result.Zvalue))

    # Restore the initial values of the parameters
    for ipar,par in enumerate(params):
        par.value = popt[ipar]

    return result

def hybrid_pvalue(pf, data, prior=None, **kw):
    """Frequentist/Bayesian hybrid hypothesis test.

    p-values are computed from the likelihood, but the nuisance parameters are
    margilanized via integration on priors::

        pvalue_cond(theta_r, theta_s) = Prob(x > x_obs | theta_r,theta_s)
        pvalue(theta_r) = integral pvalue_cond(theta_r, theta_s) * prior

    where theta_r is the parameter of interest and theta_s are the nuisance
    parameters.

    Parameters
    ----------
    pf : statspy.core.PF
        Probability function used in the computed of the likelihood
    data : ndarray, tuple
        x - variates used in the computation of the likelihood
    prior : statspy.core.PF (optional)
        Bayesian prior distribution on the nuisance parameters.
        If not specified, prior is built from parameters.
    kw : keyword arguments (optional)

        mode : str
            Can be an analytic integration 'num' or a MC based integration
            'rvs'.
        ntoys : int
            Number of toy experiments

    Returns
    -------
    result : statspy.hypotest.Result
        All information about the test is stored in the Result class.

    """
    #if prior == None:

    result = Result()
    result.pf = pf
    result.data = data
    result.prior = prior
    result.pvalue = scipy.stats.chi2.sf(result.pllr, 1)
    result.Zvalue = pvalue_to_Zvalue(result.pvalue)
