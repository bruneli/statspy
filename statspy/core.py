"""module core.py

This module contains the base classes to define a random variable,
a probability function, or a parameter.
The module is also hosting global dictonaries used to keep track of
the different variables, parameters and probability functions declared.

"""

import logging
import math
import numpy as np
import operator
import scipy.stats
import scipy.optimize

__all__ = ['RV','PF','Param','logger']

_drvs    = {}  # Dictionary hosting the list of random variables
_dpfs    = {}  # Dictionary hosting the list of probability functions
_dparams = {}  # Dictionary hosting the list of parameters

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_ch = logging.StreamHandler() # Console handler
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)
logger.debug('logger has been set to DEBUG mode')

class RV(object):
    """Base class to define a Random Variable. 

       Attributes
       ----------
       name : str
           Random Variable name
       pf : statspy.core.PF
           Probability Function object associated to a Random Variable
       params : statspy.core.Param list
           List of shape parameters used to define the pf
       isuptodate : bool
           Tells whether associated PF needs to be normalised or not
       logger : logging.Logger
           message logging system

       Examples:
       ---------
       >>> import statspy as sp 
       >>> x = sp.RV("norm(x|mu=10,sigma=2)")
    """

    def __init__(self,*args,**kwargs):
        self.name = ""
        self.pf = None
        self.params = []
        self.isuptodate = True
        self.logger = logging.getLogger('statspy.core.RV')
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            self._check_kwargs_syntax(kwargs,foundArgs)
        except:
            raise

    def pf(self,x,**kwargs):
        """Evaluate Probability (Mass/Density) Function in x

        Parameters
        ----------
        x : float, ndarray
            Random Variable value(s)
        kwargs : dictionary, optional
            Shape parameters values

        """
        if type(x) == float:
            self.logger.debug('x=%f,shape_values=%s',x,kwargs)
        return self._pf(x,**kwargs)

    def _check_args_syntax(self,args):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to PF without a keyword, it must be a string.")
        # Analyse the string
        theStr = args[0]
        if '=' in theStr:
            self.name = theStr.split('=')[0].strip().lstrip()
            self.logger.debug("Found PF name %s", self.name)
            theStr = theStr.split('=')[1]
        if not '(' in theStr:
            raise SyntaxError("No pf found in %s" % theStr)
        if not ')' in theStr:
            raise SyntaxError("Paranthesis is not closed in %s" % theStr)
        func_name = theStr.split('(')[0].strip()
        if len(func_name.split()): func_name = func_name.split()[-1]
        if not func_name in scipy.stats.__all__:
            raise SyntaxError("%s is not found in scipy.stats" % func_name)
        self.logger.debug("Found scipy.stats function named %s",func_name)
        rvNames  = theStr.split('(')[1].split(')')[0].strip().lstrip()
        parNames = rvNames
        if ';' in rvNames:
            rvNames  = rvNames.split(';')[0].strip().lstrip()
            parNames = parNames.split(';')[1].strip().lstrip()
        elif '|' in rvNames:
            rvNames  = rvNames.split('|')[0].strip().lstrip()
            parNames = parNames.split('|')[1].strip().lstrip()
        else:
            parNames = None
        lrvs = []
        for rv_name in rvNames.split(','):
            lpars.append(rv_name.strip().lstrip())
        lpars = []
        if parNames != None:
            for par_name in parNames.split(','):
                lpars.append(par_name.strip().lstrip())
        self._declare(func_name,lrvs,lpars)
        return True

    def _check_kwargs_syntax(self,kwargs,foundArgs):
        if not len(kwargs): return False
        if not foundArgs and not 'pf' in kwargs:
            raise SyntaxError("You cannot declare a Random Variable without specifying a pf.")
        if 'name' in kwargs: self.name = kwargs['name']
        if 'pf' in kwargs: self.pf = kwargs['pf']
        if 'params' in kwargs: self.params = kwargs['params']
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
        return True

    def _declare(self,pfName,rvName,lpars):
        # Set/Update Random Variable name
        self.name = rvName
        # Declare/Update parameters
        for parStr in lpars:
            parName = parStr.split('=')[0].strip().lstrip()
            parVal = 0.
            if '=' in parStr:
                parVal = float(parStr.split('=')[1].strip().lstrip())
            if not parName in _dparams:
                _dparams[parName] = {'rvs':[],'pfs':[]}
                _dparams[parName]['obj'] = Param(name=parName,value=parVal)
            if not self.name in _dparams[parName]['rvs']:
                _dparams[parName]['rvs'].append(self.name)
            self.params.append(_dparams[parName]['obj'])
        # Declare pf (no shape parameter specified yet)
        self._pf = getattr(scipy.stats,pfName)
        return

class PF(object):
    """Base class to define a Probability Function. 

       Probability Function is a generic name which includes both the
       probability mass function for discrete random variables and the
       probability density fucntion for continuous random variables.

       Attributes
       ----------
       name : str (optional)
           Function name
       func : scipy.stats.distributions.rv_generic (optional)
           Probability Density Function object associated to a Random Variable
       params : statspy.core.Param list
           List of shape parameters used to define the pf
       norm : Param
           Normalization parameter set to 1 by default. It can be different
           from 1 when the PF is fitted to data.
       isuptodate : bool
           Tells whether PF needs to be normalised or not
       logger : logging.Logger
           message logging system

       Examples
       --------
       >>> import statspy as sp
       >>> pmf_n = sp.PF("poisson(n;mu)",mu=10.)
    """

    # Define the different parameter types
    (RAW,DERIVED) = (0,10)

    def __init__(self,*args,**kwargs):
        self.name = None
        self.func = None
        self.params = []
        self.norm = Param(value=1.)
        self.isuptodate = False
        self.logger = logging.getLogger('statspy.core.PF')
        self.pftype = PF.RAW
        self._free_params = []
        self._pcov = None
        self._rvs = []
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            self._check_kwargs_syntax(kwargs,foundArgs)
            if isinstance(self.func, scipy.stats.distributions.rv_generic):
                self.isuptodate = True
        except:
            raise

    def __add__(self,other):
        """Add two PFs.

        The norm parameters are also summed.
        
        Parameters
        ----------
        self : PF
        other : PF

        Returns
        -------
        new : PF
             new pf which is the sum of self and other
        
        """
        try:
            new = PF(func=[operator.add, self, other])
        except:
            raise
        return new

    def __call__(self,*args,**kwargs):
        """Evaluate Probability Function in x

        Parameters
        ----------
        args : float, ndarray, optional, multiple values for multivariate pfs
            Random Variable value(s)
        kwargs : dictionary, optional
            Shape parameters values

        Returns
        -------
        value : float, ndarray
            Probability Function value(s) in x

        """
        # Check if self._pf contains a pdf() or a pmf() method
        if self.pftype == PF.RAW:
            method_name = 'pdf'
            try:
                if isinstance(self.func, scipy.stats.rv_discrete):
                    method_name = 'pmf'
                check_method_exists(obj=self.func,name=method_name)
            except:
                raise
        # Get random variable value(s), mandatory
        rv_values = None
        if len(args):
            rv_values = args
        else:
            rv_values = []
            for rv_name in self._rvs:
                if rv_name in kwargs: rv_values.append(kwargs[rv_name])
        if self.pftype == PF.RAW and len(rv_values) != len(self._rvs):
            raise SyntaxError('Provide %s input arguments for rvs' % 
                              len(self._rvs))
        # Get shape parameters, optional
        param_values = [0.] * len(self.params)
        for ipar,param in enumerate(self.params):
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
            param_values[ipar] = param.value
        if type(rv_values[0]) == float:
            self.logger.debug('self.func values=%s', rv_values)
        # Compute pf value in x
        #  - in case of a DERIVED PF, call an operator
        if self.pftype == PF.DERIVED:
            if not isinstance(self.func, list) or len(self.func) != 3:
                raise SyntaxError('DERIVED function is not recognized.')
            op = self.func[0]
            vals = [0.,0.]
            for idx in [1,2]:
                the_args = []
                for irv,rv_name in enumerate(self._rvs):
                    if rv_name in self.func[idx]._rvs:
                        the_args.append(rv_values[irv])
                vals[idx-1] = self.func[idx](*the_args, **kwargs)
            value = self.norm.value * op(vals[0],vals[1])
            return value
        #  - in case of a RAW PF, call directly the scipy function
        if method_name == 'pmf': 
            return self.norm.value * self.func.pmf(rv_values[0], *param_values)
        return self.norm.value * self.func.pdf(rv_values[0], *param_values)

    def __mul__(self,other):
        """Multiply a PF by another PF.
        
        parameters
        ----------
        self : PF
        other : PF

        returns
        -------
        new : PF
            new PF which is the product of self and other
        
        """
        try:
            new = PF(func=[operator.mul, self, other])
        except:
            raise
        return new

    def dF(self, x):
        """Compute the uncertainty on PF given the uncertainty on the shape
        and norm parameters.

        This method can be used to show an error band on your fitted PF.
        To compute the uncertainty on the PF, the error propagation formula is
        used,
            dF(x;th) = (F(x;th+dth) - F(x;th-dth))/2
            dF(x)^2 = dF(x;th)^T * corr(th,th') * dF(x;th')
        so keep in mind it is only an approximation.

        parameters
        ----------
        x : float, ndarray
            Random variate(s)

        returns
        -------
        dF : float, ndarray
            Uncertainty on the PF evaluated in x

        """
        # Get list of free parameters
        self.get_list_free_params()
        npars = len(self._free_params)
        if npars == 0: return np.zeros(len(x))
        popt = np.ndarray(npars)
        punc = np.ndarray(npars)
        for ipar,par in enumerate(self._free_params):
            popt[ipar] = par.value
            punc[ipar] = par.unc
        # Build the correlation matrix
        corr = np.ndarray((npars, npars))
        if self._pcov != None:
            if self._pcov.shape[0] != npars or self._pcov.shape[1] != npars:
                raise SyntaxError('covariance matrix is not defined properly')
            for ipar in range(npars):
                for jpar in range(npars):
                    corr[ipar][jpar] = self._pcov[ipar][jpar]
                    if punc[ipar] != 0.: corr[ipar][jpar] /= punc[ipar]
                    if punc[jpar] != 0.: corr[ipar][jpar] /= punc[jpar]
        else:
            corr = np.diag(np.ones(npars))
        mcorr = np.asmatrix(corr)
        # Compute dF(x;th)
        if not isinstance(x, np.ndarray): x = np.asarray([x])
        pf_plus  = None
        pf_minus = None
        for ipar in range(npars):
            # theta -> theta + deltaTheta
            (self._free_params[ipar]).value = popt[ipar] + punc[ipar]
            y = self(x)
            if pf_plus == None:
                pf_plus = y
            else:
                pf_plus = np.vstack((pf_plus, y))
            # theta -> theta - deltaTheta
            (self._free_params[ipar]).value = popt[ipar] - punc[ipar]
            y = self(x)
            if pf_minus == None:
                pf_minus = y
            else:
                pf_minus = np.vstack((pf_minus, y))
        dF_th = np.asmatrix(0.5 * (pf_plus - pf_minus))
        # Compute dF
        dF = (dF_th.T * mcorr) * dF_th
        return np.sqrt(np.diag(dF))

    def get_list_free_params(self):
        """Get the list of free parameters."""
        self._free_params = []
        # Get the list of normalization factors
        if not self.norm.const and self.norm.partype == Param.RAW:
            self._free_params.append(self.norm)
        if self.pftype == PF.DERIVED and type(self.func) == list:
            for ele in self.func:
                if not isinstance(ele, PF): continue
                if ele.norm.partype == Param.RAW and not ele.norm.const:
                    self._free_params.append(ele.norm)
                elif ele.norm.partype == Param.DERIVED:
                    raw_params = ele.norm.get_raw_params()
                    for raw_par in raw_params:
                        if raw_par.const: continue
                        self._free_params.append(raw_par)
        # Get the list of shape parameters
        for par in self.params:
            if par.partype == Param.RAW and not par.const:
                self._free_params.append(par)
            elif par.partype == Param.DERIVED:
                raw_params = par.get_raw_params()
                for raw_par in raw_params:
                    if raw_par.const: continue
                    self._free_params.append(raw_par)
        return self._free_params

    def leastsq_fit(self, xdata, ydata, ey=None, dx=None, **kw):
        """Fit the PF to data using a least squares method.

        The fitting part is performed using the scipy.optimize.leastsq 
        function. The Levenberg-Marquardt algorithm is used by the 'leastsq'
        method to find the minimum values.
        When calling this method, all PF parameters are minimized except
        the one which are set as 'const'.

        Parameters
        ----------
        xdata : ndarray
            Values for which ydata are measured and PF must be computed
        ydata : ndarray
            Observed values (like number of events)
        ey : ndarray (optional)
            Standard deviations of ydata. If not specified, it takes
            sqrt(ydata) as standard deviation.
        dx : ndarray (optional)
            Array containing bin-width of xdata. It can be used to normalize
            the PF to the integral while minimizing.
        kw : keyword arguments
            Keyword arguments passed to the leastsq method

        Returns
        -------
        free_params : statspy.core.Param list
             List of the free parameters used during the fit. Their 'value'
             and 'unc' arguments are extracted from minimization.
        pcov : 2d array
             Estimated covariance matrix of the free parameters.
        chi2min : float
             Least square sum evaluated in popt.
        pvalue : float
             p-value = P(chi2>chi2min,ndf) with P a chi2 distribution and
             ndf the number of degrees of freedom.

        """
        # Define parameters which should be minimized and set initial values
        self.get_list_free_params()
        p0 = np.ones(len(self._free_params))
        for ipar,par in enumerate(self._free_params):
             p0[ipar] = par.value
        # Compute weights
        if ey == None: ey = np.sqrt(ydata)
        ey[ey == 0] = 1.
        weight = 1./np.asarray(ey)
        # Call the leastsq method
        if dx == None: dx = np.ones(xdata.shape)
        args = (xdata, ydata, weight, dx)
        res = scipy.optimize.leastsq(self._leastsq_function, p0,
                                     args=args, full_output=1, **kw)
        # Manage results
        (popt, self._pcov, infodict, errmsg, ier) = res
        if ier not in [1,2,3,4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)
        chi2min = (self._leastsq_function(popt, *args)**2).sum()
        if (len(ydata) > len(p0)) and self._pcov is not None:
            ndf = len(ydata)-len(p0)
            self._pcov = self._pcov * chi2min / ndf
            pvalue = scipy.stats.chi2.sf(chi2min, ndf)
            for ipar,par in enumerate(self._free_params):
                if (self._pcov)[ipar][ipar] >= 0.:
                    par.unc = math.sqrt((self._pcov)[ipar][ipar])
        else:
            self._pcov = inf
            pvalue = inf
        for ipar,par in enumerate(self._free_params):
            par.value = popt[ipar]
        return self._free_params, self._pcov, chi2min, pvalue

    def rvs(self, **kwargs):
        """Get random variates from a PF

        Keyword arguments
        -----------------
        size : int
             Number of random variates
        mu, sigma,... : float
             Any parameter name used while declaring the PF

        Returns
        -------
        data : ndarray
             Array of random variates

        Examples
        --------
        >>> import statspy as sp
        >>> pdf_x = sp.PF("pdf_x=norm(x;mu=20,sigma=5)")
        >>> data = pdf_x.rvs(size=1000)

        """
        try:
            method_name = "rvs"
            check_method_exists(obj=self.func,name=method_name)
            shape_params = []
            for param in self.params:
                if param.name in kwargs:
                    param.value = kwargs[param.name]
                shape_params.append(param.value)
            data = self.func.rvs(*shape_params, **kwargs)
        except:
            raise
        return data

    def _check_args_syntax(self,args):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to PF without a keyword, it must be a string.")
        # Analyse the string
        theStr = args[0]
        if not '(' in theStr:
            raise SyntaxError("No pf found in %s" % theStr)
        if not ')' in theStr:
            raise SyntaxError("Paranthesis is not closed in %s" % theStr)
        func_name = theStr.split('(')[0].strip()
        if '=' in func_name:
            self.name = func_name.split('=')[0].strip().lstrip()
            self.norm.name = 'norm_%s' % self.name
            self.logger.debug("Found PF name %s", self.name)
            func_name = func_name.split('=')[1]
        if len(func_name.split()): func_name = func_name.split()[-1]
        if not func_name in scipy.stats.__all__:
            raise SyntaxError("%s is not found in scipy.stats" % func_name)
        self.logger.debug("Found scipy.stats function named %s",func_name)
        rvNames  = theStr.split('(')[1].split(')')[0].strip().lstrip()
        parNames = rvNames
        if ';' in rvNames:
            rvNames  = rvNames.split(';')[0].strip().lstrip()
            parNames = parNames.split(';')[1].strip().lstrip()
        elif '|' in rvNames:
            rvNames  = rvNames.split('|')[0].strip().lstrip()
            parNames = parNames.split('|')[1].strip().lstrip()
        else:
            parNames = None
        for rv_name in rvNames.split(','):
            if not rv_name in self._rvs: self._rvs.append(rv_name)
        lpars = []
        if parNames != None:
            for par_name in parNames.split(','):
                lpars.append(par_name.strip().lstrip())
        self._declare(func_name,lpars)
        return True

    def _check_kwargs_syntax(self,kwargs,foundArgs):
        if not len(kwargs): return False
        if 'name' in kwargs:
            if self.name != None:
                raise SyntaxError("self.name is already set to %s" % self.name)
            self.name = kwargs['name']
            self.norm.name = 'norm_%s' % self.name
        if not foundArgs and not 'func' in kwargs:
            raise SyntaxError("You cannot declare a PF without specifying a function to caracterize it.")
        if 'func' in kwargs:
            if self.func != None:
                raise SyntaxError("self.func already exists.")
            self.func = kwargs['func']
            if type(self.func) == list:
                self.pftype = PF.DERIVED
                for ele in self.func:
                    if not isinstance(ele, PF): continue
                    for par in ele.params:
                        if not par in self.params: self.params.append(par)
                    for rv_name in ele._rvs:
                        if not rv_name in self._rvs: self._rvs.append(rv_name)
            else:
                self.pftype = PF.RAW
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
        return True

    def _declare(self,func_name,lpars):
        # Declare functional form of PF (no shape parameter specified yet)
        self.func = getattr(scipy.stats,func_name)
        self.pftype = PF.RAW
        # Declare/Update parameters
        for parStr in lpars:
            parName = parStr.split('=')[0].strip().lstrip()
            parVal = 0.
            if '=' in parStr:
                parVal = float(parStr.split('=')[1].strip().lstrip())
            if not parName in _dparams:
                _dparams[parName] = {'rvs':[],'pfs':[]}
                _dparams[parName]['obj'] = Param(name=parName,value=parVal)
            if self.name != None:
                if not self.name in _dparams[parName]['pfs']:
                    _dparams[parName]['pfs'].append(self.name)
            else:
                if not self in _dparams[parName]['pfs']:
                    _dparams[parName]['pfs'].append(self)
            self.params.append(_dparams[parName]['obj'])
        return

    def _leastsq_function(self, params, xdata, ydata, weight, dx):
        """Function used by scipy.optimize.leastsq"""
        # Update values of non-const PF parameters
        for ipar,par in enumerate(self._free_params):
            par.value = params[ipar]
        # Return delta = (PF(x) - y)/sigma
        return weight * (self(xdata) * dx - ydata)

class Param(object):
    """Base class to define a PF shape parameter. 

       Two types of parameters can be built:
       - RAW parameters do not depend on any other parameter and have
       a value directly associated to them
       - DERIVED parameters are obtained from other parameters via an 
       analytical formula.

       Attributes
       ----------
       name : str
           Random Variable name
       value : float
           Current numerical value
       unc : float
           Parameter uncertainty (e.g. after minimization)
       bounds : list
           Defines, if necessary, the lower and upper bounds
       formula : list (optional, only for DERIVED parameters)
           List of operators and parameters used to parse an analytic function
       strform : str (optional, only for DERIVED parameters)
           Representation of the formula as a string
       partype : int
           Tells whether it is a RAW or a DERIVED parameter
       const : bool
           Tells whether a parameter is fixed during a minimazation process.
           It is not a constant in the sense of C++.
       isuptodate : bool
           Tells whether value needs to be computed again or not
       logger : logging.Logger
           message logging system

       Examples
       --------
       >>> import statspy as sp 
       >>> mu = sp.Param(name="mu",value=10.)
    """

    # Define the different parameter types
    (RAW,DERIVED) = (0,10)

    def __init__(self,*args,**kwargs):
        self.name    = kwargs.get('name', None)
        self.value   = kwargs.get('value', 0.)
        self.unc     = kwargs.get('unc', 0.)
        self.bounds  = kwargs.get('bounds', [])
        self.formula = kwargs.get('formula', None)
        self.strform = kwargs.get('strform', None)
        self.const   = kwargs.get('const', False)
        self.partype = Param.RAW
        self.isuptodate = True
        self.logger  = logging.getLogger('statspy.core.Param')
        try:
            if self.formula != None:
                self.partype = Param.DERIVED
                #Param._check_formula(self.formula) # <= TODO
                Param._update_db(self, self)
                self._evaluate()
            if self.name != None:
                self._register_in_db(self.name)
        except:
            raise

    def __add__(self,other):
        """Add a parameter to another parameter or a numerical value.
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        new : Param
             new parameter which is the sum of self and other
        
        """
        try:
            new = Param(formula=[[operator.add, self, other]],
                        strform=Param._build_str_formula(self,'+',other))
        except:
            raise
        return new

    def __call__(self):
        """Return the parameter value.

        Returns
        -------
        self.value : float
            Parameter value possibly recomputed from self.formula
            if self.isuptodate is False.
        """
        return self.value

    def __div__(self,other):
        """Divide a parameter by another parameter or by a numerical value.
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        new : Param
             new parameter which is the ratio of self and other
        
        """
        try:
            new = Param(formula=[[operator.div, self, other]],
                        strform=Param._build_str_formula(self,'/',other))
        except:
            raise
        return new

    def __getattribute__(self, name):
        """Overload __getattribute__ to update the value attribute from the 
        formula for a DERIVED parameter.
        """
        try:
            if (name == "value" and self.partype == Param.DERIVED and
                (self.name == None or self.isuptodate == False)):
                self._evaluate()
        except:
            raise
        return object.__getattribute__(self, name)

    def __mul__(self,other):
        """Multiply a parameter by another parameter or by a numerical value.
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        new : Param
             new parameter which is the product of self and other
        
        """
        try:
            new = Param(formula=[[operator.mul, self, other]],
                        strform=Param._build_str_formula(self,'*',other))
        except:
            raise
        return new

    def __pow__(self,other):
        """Raise a parameter the power.
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        new : Param
             new parameter which is self raised to the power of other
        
        """
        try:
            new = Param(formula=[[operator.pow, self, other]],
                        strform=Param._build_str_formula(self,'**',other))
        except:
            raise
        return new

    def __radd__(self,other):
        """Add a numerical value to a parameter"""
        try:
            new = Param(formula=[[operator.add, other, self]],
                        strform=Param._build_str_formula(other,'+',self))
        except:
            raise
        return new

    def __rdiv__(self,other):
        """Divide a numerical value by a parameter"""
        try:
            new = Param(formula=[[operator.div, other, self]],
                        strform=Param._build_str_formula(other,'/',self))
        except:
            raise
        return new

    def __repr__(self):
        """Return Parameter value and formula if DERIVED"""
        theStr = self.name + " = " if self.name != None else ""
        if self.strform != None: theStr = theStr + self.strform + " = "
        theStr = theStr + str(self.value)
        return theStr

    def __rmul__(self,other):
        """Multiply a numerival value by a parameter"""
        try:
            new = Param(formula=[[operator.mul, other, self]],
                        strform=Param._build_str_formula(other,'*',self))
        except:
            raise
        return new

    def __rpow__(self,other):
        """Raise a numerical value to the power"""
        try:
            new = Param(formula=[[operator.pow, other, self]],
                        strform=Param._build_str_formula(other,'**',self))
        except:
            raise
        return new

    def __rsub__(self,other):
        """Subtract a numerical value to a parameter"""
        try:
            new = Param(formula=[[operator.sub, other, self]],
                        strform=Param._build_str_formula(other,'-',self))
        except:
            raise
        return new

    def __setattr__(self, name, value):
        """Overload __setattr__ to make sure that quantities based on this
        parameter will be updated when its value is modified.
        """
        if "value" in self.__dict__ and name == "value":
            if hasattr(self, "name") and self.name != None:
                update_status(_dparams[self.name])
        if name == "name" and hasattr(self, "name"):
            if self.name != None:
                raise AttributeError('Cannot overwrite parameter name')
            self._register_in_db(value)
        super(Param, self).__setattr__(name, value)

    def __sub__(self,other):
        """Subtract a parameter to another parameter or a numerical value.
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        new : Param
             new parameter which is the difference of self and other
        
        """
        try:
            new = Param(formula=[[operator.sub, self, other]])
        except:
            raise
        return new

    def get_raw_params(self):
        raw_params = []
        if self.partype != Param.DERIVED: return raw_params
        if self.func == None or type(self.func) != list:
            return raw_params
        for ele in self.func:
            if not isinstance(ele, Param): continue
            if ele.partype == Param.RAW:
                raw_params.append(ele)
            elif ele.partype == Param.DERIVED:
                raw_params += ele.get_raw_params()
        return raw_params

    def _evaluate(self):
        if self.partype != Param.DERIVED: return
        if type(self.formula) != list: return
        for op in self.formula:
            if isinstance(op, list):
                if len(op) == 3:
                    val1 = op[1].value if isinstance(op[1], Param) else op[1]
                    val2 = op[2].value if isinstance(op[2], Param) else op[2]
                    self.value = op[0](val1, val2)
                elif len(op) == 2 and isinstance(op[1], Param):
                    self.value = op[0](op[1].value)
                elif len(op) == 1 and isinstance(op[0], Param):
                    self.value = op[0].value
                else:
                    raise TypeError('operation is not recognized')
            elif isinstance(op, Param):
                self.value = op.value
            else:
                raise TypeError('operation type is not recognized')
        self.isuptodate = True
        return

    def _register_in_db(self, name):
        """Register a parameter in database."""
        if name in _dparams and 'obj' in _dparams[name]:
            self.logger.warning('%s already registred, remove existing info',
                                name)
        _dparams[name] = {'rvs':[],'pfs':[],'params':[]}
        _dparams[name]['obj'] = self
        self.logger.debug('Register new Param: %s with value=%f, bounds=%s',
                          name,self.value,self.bounds)
        if self.formula != None:
            self.logger.debug('and derived from formula: %s', self.formula)
        return

    @staticmethod
    def _update_db(par, par_to_add):
        """Add par_to_add to the list of params in database"""
        if par.partype == Param.RAW:
            if par.name != None:
                _dparams[par.name]['params'].append(par_to_add)
            return
        if not isinstance(par.formula, list) or not len(par.formula): return
        for op in par.formula:
            if isinstance(op, list):
                for ele in op:
                    if not isinstance(ele, Param): continue
                    if ele.name == None:
                        if ele.partype == Param.DERIVED:
                            Param._update_db(ele, par_to_add)
                    else:
                        _dparams[ele.name]['params'].append(par_to_add)
            elif isinstance(op, Param):
                if op.name == None:
                    if op.partype == Param.DERIVED:
                        Param._update_db(op, par_to_add)
                else:
                    _dparams[op.name]['params'].append(par_to_add)
            else:
                raise TypeError('operation is not recognized')
        return

    @staticmethod
    def _get_strform(par):
        strform = None
        if isinstance(par, (int,long,float)): return str(par)
        if par.name != None:
            strform = par.name
        elif par.strform != None:
            strform = par.strform
        return strform

    @staticmethod
    def _build_str_formula(par1, op, par2):
        strform = None
        par1_strform = Param._get_strform(par1)
        par2_strform = Param._get_strform(par2)
        if par1_strform != None and par2_strform != None:
            if op == '+' or op == '-':
                strform = '%s %s %s' % (par1_strform, op, par2_strform)
            else:
                strform = '(%s) %s (%s)' % (par1_strform, op, par2_strform)
        return strform

def check_method_exists(obj=None,name=""):
    if obj == None:
        raise StandardError('Object is not defined, check syntax.')
    if (not hasattr(obj,name) or 
        not callable(getattr(obj,name))):
        raise StandardError('No %s() method found.' % name)
    return True
 
def update_status(obj_dict):
    """Check the list of objects depending on this parameter and update
    their isuptodate status to False.
    """
    for obj_key in ['rvs','pfs','params']:
        if not obj_key in obj_dict: continue
        for obj in obj_dict[obj_key]:
            if hasattr(obj, "isuptodate"): obj.isuptodate = False
            if hasattr(obj, "name") and obj.name != None:
                if obj_key == "rvs":
                    update_status(_drvs[obj.name])
                if obj_key == "pfs":
                    update_status(_dpfs[obj.name])
                if obj_key == "params":
                    update_status(_dparams[obj.name])
