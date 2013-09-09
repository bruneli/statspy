"""module interval.py

This module contains functions to estimate confidence or credible intervals.

"""

import logging
import math
import scipy.optimize
import scipy.stats

__all__ =['pllr']

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_ch = logging.StreamHandler() # Console handler
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)
logger.debug('logger has been set to DEBUG mode')

def pllr(pf, data, **kw):
    """Compute confidence intervals using a profile likelihood ratio method.

    Interval estimation is done through steps:

    * The (minus log-)likelihood is computed from ``pf`` and ``data`` via the
      ``PF.nllf`` method. 
    * Best estimates \hat{theta_i} for each parameter theta_i are computed with
      the ``PF.maxlikelihood_fit`` method.
    * The confidence interval of theta_i around its best estimate is computed
      from a profile log-likelihood ratio function q defined as::

        l = L(x|theta_i,\hat{\hat{theta_s}}) / L(x|\hat{theta_i},\hat{theta_s})
        q(theta_i) = -2 * log(l)

      where L is the likelihood function and theta_s are the nuisance 
      parameters.
    * q(theta_i) is assumed to be described as a chi2 distribution (Wilks'
      theorem). Bounds corresponding to a given confidence level (CL) are found
      by searching values for which q(theta_i) is equal to the chi2 quantile 
      of CL::

        quantile = scipy.stats.chi2.ppf(cl, ndf)

    Parameters
    ----------
    pf : statspy.core.PF
        Probability function used in the computation of the likelihood
    data : ndarray, tuple
        x - variates used in the computation of the likelihood 
    kw : keyword arguments (optional)
        Possible keyword arguments are:

        cl : float
            Confidence level (0.6827 by default)
        ndf : int
            Number of degrees of freedom (1 by default)
        root_finder : scipy.optimize function
            Root finder algorithm (scipy.optimize.brentq by default)

    Returns
    -------
    params : statspy.core.Param list
        List of the parameters for which a confidence interval has been
        extracted including updated 'value', 'neg_unc' and 'pos_unc' arguments.
    corr : ndarray
        Correlation matrix
    quantile : float
        Quantile used in the computation of bounds

    """
    # Confidence level and corresponding quantile
    cl  = kw.get('cl', 0.6827)
    ndf = kw.get('ndf', 1)
    quantile = scipy.stats.chi2.ppf(cl, ndf)
    logger.debug('confidence level = %f, ndf = %d, quantile = %f' % 
                 (cl, ndf, quantile))

    # Algorithm used to find roots of (q - quantile = 0)
    root_finder = kw.get('root_finder', scipy.optimize.brentq)

    # Maximum likelihood estimates
    params, nllfmin = pf.maxlikelihood_fit(data, method='BFGS')
    popt = []
    [ popt.append([par.value, par.unc]) for par in params ]
    corr = pf.corr()

    # Loop over parameters
    maxiter = kw.get('maxiter', 20)
    for ipar,par in enumerate(params):
        # set par as the only parameter of interest
        for ipar2,par2 in enumerate(params):
            par2.poi = False
            par2.value = popt[ipar2][0]
            par2.unc = popt[ipar2][1]
        par.poi = True
        logger.debug('%s max likelihood estimate = %f' % (par.name, par.value))
        logger.debug('%s quadratic uncertainty = %f' % (par.name, par.unc))
        # upper bound
        iter = 0
        range = [popt[ipar][0], 
                 popt[ipar][0] + math.sqrt(quantile) * popt[ipar][1]]
        par.value = range[1]
        while (pf.pllr(data, uncond_nllf=nllfmin) < quantile and 
               iter < maxiter):
            range[1] = range[1] + math.sqrt(quantile) * popt[ipar][1]
            par.value = range[1]
            iter += 1
        par.value = range[0]
        logger.debug('%s upper bound root finding range is %s' % (par.name, range))
        upper_bound = root_finder(_pllr_root_finding, range[0], range[1],
                                  args=(pf, data, par, quantile, nllfmin))
        logger.debug('%s upper bound = %f' % (par.name, upper_bound))
        # lower bound
        iter = 0
        range = [popt[ipar][0] - math.sqrt(quantile) * popt[ipar][1],
                 popt[ipar][0]]
        par.value = range[0]
        while (pf.pllr(data, uncond_nllf=nllfmin) < quantile and iter < maxiter):
            range[0] = range[0] - math.sqrt(quantile) * popt[ipar][1]
            par.value = range[0]
            iter += 1
        par.value = range[1]
        logger.debug('%s lower bound root finding range is %s' % (par.name, range))
        lower_bound = root_finder(_pllr_root_finding, range[0], range[1],
                                  args=(pf, data, par, quantile, nllfmin))
        logger.debug('%s lower bound = %f' % (par.name, lower_bound))
        #
        par.neg_unc = lower_bound - popt[ipar][0]
        par.pos_unc = upper_bound - popt[ipar][0]

    for ipar,par in enumerate(params):
        par.value = popt[ipar][0]
        par.unc = popt[ipar][1]
    return params, corr, quantile

def _pllr_root_finding(x, pf, data, par, quantile, nllfmin):
    par.value = x
    return (pf.pllr(data, uncond_nllf=nllfmin) - quantile)
