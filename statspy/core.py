"""module core.py

This module contains the base classes to define a random variable,
a probability function, or a parameter.
The module is also hosting global dictonaries used to keep track of
the different variables, parameters and probability functions declared.

"""

import copy
import logging
import math
import numpy as np
import operator
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.stats

__all__ = ['Param','PF','RV','get_obj','exp','log','sqrt']

_dparams = {}  # Dictionary hosting the list of parameters
_dpfs    = {}  # Dictionary hosting the list of probability functions
_drvs    = {}  # Dictionary hosting the list of random variables

# Logging system
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
_ch = logging.StreamHandler() # Console handler
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch.setFormatter(_formatter)
logger.addHandler(_ch)
logger.debug('logger has been set to DEBUG mode')

class Param(object):
    """Base class to define a PF shape parameter. 

       Two types of parameters can be built:

       * RAW parameters do not depend on any other parameter and have
         a value directly associated to them
       * DERIVED parameters are obtained from other parameters via an 
         analytical formula.

       Attributes
       ----------
       name : str
           Random Variable name
       label : str
           Random Variable name for printing purposes
       value : float
           Current numerical value
       unc : float
           Parameter uncertainty (e.g. after minimization)
       neg_unc : float
           Negative parameter uncertainty (only for intervals)
       pos_unc : float
           Positive parameter uncertainty (only for intervals)
       bounds : list
           Defines, if necessary, the lower and upper bounds
       formula : list (optional, only for DERIVED parameters)
           List of operators and parameters used to parse an analytic function
       strform : str (optional, only for DERIVED parameters)
           Representation of the formula as a string
       partype : Param.RAW or Param.DERIVED
           Tells whether it is a RAW or a DERIVED parameter
       const : bool
           Tells whether a parameter is fixed during a minimazation process.
           It is not a constant in the sense of programming.
       poi :  bool
           Tells whether a parameter is a parameter of interest in an
           hypothesis test.
       isuptodate : bool
           Tells whether value needs to be computed again or not
       logger : logging.Logger
           message logging system

       Examples
       --------

       Declarations::

           >>> import statspy as sp 
           >>> mu = sp.Param(name="mu",value=10.,label="\\\\mu")
           >>> x = sp.Param("x = 10. +- 2.")
           >>> x.value
           10.0
           >>> x.unc
           2.0
           >>> y = sp.Param("y = 5.", unc=1.)

       Operations, building DERIVED parameters::

           >>> x + y
           x + y = 15.0 +- 2.2360679775
           >>> z = x * y
           >>> z.name = 'z'
           >>> z
           z = x * y = 50.0 +- 14.1421356237
           >>> x**2
           x ** 2 = 100.0 +- 40.0

       Possible operations are ``+,-,*,/,**``.

    """

    # Define the different parameter types
    (RAW,DERIVED) = (0,10)

    def __init__(self, *args, **kwargs):
        # Public class members
        self.name    = kwargs.get('name', None)
        self.label   = kwargs.get('label', self.name)
        self.value   = kwargs.get('value', 0.)
        self.unc     = kwargs.get('unc', 0.)
        self.neg_unc = kwargs.get('neg_unc', 0.)
        self.pos_unc = kwargs.get('pos_unc', 0.)
        self.bounds  = kwargs.get('bounds', [])
        self.formula = kwargs.get('formula', None)
        self.strform = kwargs.get('strform', None)
        self.const   = kwargs.get('const', False)
        self.poi     = kwargs.get('poi', False)
        self.partype = Param.RAW
        self.isuptodate = True
        self.logger  = logging.getLogger('statspy.core.Param')
        # Internal class members
        self._derivative = kwargs.get('derivative', None)
        self._pcov = None
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            if self.formula != None:
                self.partype = Param.DERIVED
                #Param._check_formula(self.formula) # <= TODO
                Param._update_db(self, self)
                self._evaluate()
            if self.name != None:
                self._register_in_db(self.name)
        except:
            raise

    def __add__(self, other):
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

    def __div__(self, other):
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
            if (name == "unc" and self.partype == Param.DERIVED):
                self._evaluate_unc()
        except:
            raise
        return object.__getattribute__(self, name)

    def __iadd__(self, other):
        """In-place addition (+=)
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        self : Param
             self parameter modified by other
        
        """
        try:
            if self.partype == Param.RAW:
                self.partype = Param.DERIVED
                self.isuptodate = False
                self.formula = []
            self.formula.append([operator.add, self, other])
            self.strform = Param._build_str_formula(self,'+',other)
        except:
            raise
        return self

    def __imul__(self, other):
        """In-place multiplication (*=)
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        self : Param
             self parameter modified by other
        
        """
        try:
            if self.partype == Param.RAW:
                self.partype = Param.DERIVED
                self.isuptodate = False
                self.formula = []
            self.formula.append([operator.mul, self, other])
            self.strform = Param._build_str_formula(self,'*',other)
        except:
            raise
        return self

    def __isub__(self, other):
        """In-place subtraction (-=)
        
        Parameters
        ----------
        self : Param
        other : Param, int, long, float

        Returns
        -------
        self : Param
             self parameter modified by other
        
        """
        try:
            if self.partype == Param.RAW:
                self.partype = Param.DERIVED
                self.isuptodate = False
                self.formula = []
            self.formula.append([operator.sub, self, other])
            self.strform = Param._build_str_formula(self,'-',other)
        except:
            raise
        return self

    def __mul__(self, other):
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

    def __pow__(self, other):
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

    def __radd__(self, other):
        """Add a numerical value to a parameter"""
        try:
            new = Param(formula=[[operator.add, other, self]],
                        strform=Param._build_str_formula(other,'+',self))
        except:
            raise
        return new

    def __rdiv__(self, other):
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
        if self.unc != 0: theStr = theStr + " +- " + str(self.unc)
        return theStr

    def __rmul__(self, other):
        """Multiply a numerival value by a parameter"""
        try:
            new = Param(formula=[[operator.mul, other, self]],
                        strform=Param._build_str_formula(other,'*',self))
        except:
            raise
        return new

    def __rpow__(self, other):
        """Raise a numerical value to the power"""
        try:
            new = Param(formula=[[operator.pow, other, self]],
                        strform=Param._build_str_formula(other,'**',self))
        except:
            raise
        return new

    def __rsub__(self, other):
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

    def __sub__(self, other):
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
            new = Param(formula=[[operator.sub, self, other]],
                        strform=Param._build_str_formula(self,'-',other))
        except:
            raise
        return new

    def get_raw_params(self):
        """Get the list of RAW parameters from a DERIVED parameter"""
        raw_params = []
        if self.partype != Param.DERIVED: return raw_params
        if self.formula == None or type(self.formula) != list:
            return raw_params
        for op in self.formula:
            for ele in op:
                if not isinstance(ele, Param): continue
                if ele.partype == Param.RAW:
                    raw_params.append(ele)
                elif ele.partype == Param.DERIVED:
                    raw_params += ele.get_raw_params()
        return raw_params

    def unbound_repr(self):
        """Transform the value of the parameter such as it has no bounds.

        This method is used with the minimization algorithms which require
        only unbound parameters. The transformation formula from a
        double-sided or a one-sided parameter to an unbound parameter are 
        described in Section 1.3.1 of the MINUIT manual:
        Fred James, Matthias Winkler, MINUIT User's Guide, June 16, 2004.

        Returns
        -------
        new : float
            parameter value within an unbound representation.

        """
        new = self.value
        if len(self.bounds) != 2: return new # Parameter has no bounds
        a = self.bounds[0]
        b = self.bounds[1]
        if a != None and b != None:
            # Parameter with double-sided limits
            new = math.asin(2 * (new - a) / (b - a) - 1)
        elif a != None:
            # Parameter with a lower bound
            new = math.sqrt((new - a + 1) * (new - a + 1) - 1)
        elif b != None:
            # Parameter with an upper bound
            new = math.sqrt((b - new + 1) * (b - new + 1) - 1)
        return new

    def unbound_to_bound(self, val, unc = 0.):
        """Transform the parameter value from an unbound to a bounded 
        representation.

        This method is used with the minimization algorithms which require
        only unbound parameters. The transformation formula from an unbound
        parameter to a double-sided or a one-sided parameter are described in
        Section 1.3.1 of the MINUIT manual:
        Fred James, Matthias Winkler, MINUIT User's Guide, June 16, 2004.
        Since the transformation is non-linear, the transformation of the
        uncertainty is approximate and based on the error propagation 
        formula. In particular, when the value is close to its limit, the
        uncertainty is not trustable and a more refined analysis should be 
        performed.

        Parameters
        ----------
        val : float
            parameter value within an unbound representation
        unc : float
            parameter uncertainty within an unbound representation

        """
        if len(self.bounds) != 2:
            self.value = val
            self.unc   = unc
            return self.value, self.unc # Parameter has no bounds
        a = self.bounds[0]
        b = self.bounds[1]
        if a != None and b != None:
            # Parameter with double-sided limits
            self.value = a + (b - a) / 2 * (math.sin(val) + 1)
            self.unc   = math.fabs((b - a) / 2 * math.cos(val) * unc)
        elif a != None:
            # Parameter with a lower bound
            self.value = a - 1 + math.sqrt(val * val + 1)
            self.unc   = val / math.sqrt(val * val + 1) * unc
        elif b != None:
            # Parameter with an upper bound
            self.value = b + 1 - math.sqrt(val * val + 1)
            self.unc   = val / math.sqrt(val * val + 1) * unc
        return self.value, self.unc

    def _check_args_syntax(self, args):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to Param without a keyword, it must be a string.")
        # Analyse the string
        theStr = args[0]
        if '=' in theStr:
            self.label = self.name = theStr.split('=')[0].strip().lstrip()
            self.logger.debug("Found Param name %s", self.name)
            theStr = theStr.split('=')[1]
        if '+-' in theStr:
            self.unc = float(theStr.split('+-')[1].strip().lstrip())
            theStr = theStr.split('+-')[0]
        self.value = float(theStr.strip().lstrip())
        return True

    def _evaluate(self):
        if self.partype != Param.DERIVED: return
        if type(self.formula) != list: return
        value = 0.
        for op in self.formula:
            if isinstance(op, list):
                if len(op) == 3:
                    if isinstance(op[1], Param):
                        val1 = value if op[1] == self else op[1].value 
                    else:
                        val1 = op[1]
                    val2 = op[2].value if isinstance(op[2], Param) else op[2]
                    value = op[0](val1, val2)
                elif len(op) == 2 and isinstance(op[1], Param):
                    if op[1] == self:
                        value = op[0](value)
                    else:
                        value = op[0](op[1].value)
                elif len(op) == 1 and isinstance(op[0], Param):
                    value = op[0].value
                else:
                    raise TypeError('operation is not recognized')
            elif isinstance(op, Param):
                value = op.value
            else:
                raise TypeError('operation type is not recognized')
            self.value = value
        self.isuptodate = True
        return

    def _evaluate_unc(self):
        """Propagation of uncertainty."""
        if self.partype != Param.DERIVED: return
        if type(self.formula) != list: return
        value = 0.
        unc = 0.
        for op in self.formula:
            if isinstance(op, list):
                if len(op) == 3:
                    # operators +,-,*,/,**
                    if op[0] == operator.mul:
                        derivative = [lambda x,y : y, lambda x,y : x]
                    elif op[0] == operator.div:
                        derivative = [lambda x,y : 1/y, lambda x,y : x/y/y]
                    elif op[0] == operator.pow:
                        derivative = [lambda x,y : y*(x**(y-1)), 
                                      lambda x,y : math.log(x)*math.exp(y) ]
                    else:
                        derivative = [lambda x,y : 1, lambda x,y : 1]
                    corr = 0.
                    if isinstance(op[1], Param):
                        val1 = value if op[1] == self else op[1].value
                        unc1 = unc if op[1] == self else op[1].unc
                        if type(op[1]._pcov) == list and isinstance(op[2], Param):
                            for par in op[1]._pcov:
                                if par[0] == op[2]: corr = par[1] 
                    else:
                        val1 = op[1]
                        unc1 = 0.
                    val2 = op[2].value if isinstance(op[2], Param) else op[2]
                    unc2 = op[2].unc if isinstance(op[2], Param) else 0.
                    df1 = (derivative[0])(val1, val2)
                    df2 = (derivative[1])(val1, val2)
                    value = op[0](val1, val2)
                    unc = df1 * df1 * unc1 * unc1 + df2 * df2 * unc2 * unc2
                    if corr != 0.: unc += df1 * df2 * corr * unc1 * unc2
                    if unc < 0.: raise SyntaxError('Negative uncertainty.')
                    unc = math.sqrt(unc)
                elif len(op) == 2 and isinstance(op[1], Param):
                    # mathematical function
                    if op[1] == self:
                        value = op[0](value)
                    else:
                        value = op[0](op[1].value)
                        if self._derivative != None:
                            df = self._derivative(op[1].value)
                            unc = df * df * op[1].unc * op[1].unc
                            if unc < 0.: raise SyntaxError('Negative uncertainty.')
                            unc = math.sqrt(unc)
                elif len(op) == 1 and isinstance(op[0], Param):
                    # simple copy
                    value = op[0].value
                    unc = op[0].unc
                else:
                    raise TypeError('operation is not recognized')
            elif isinstance(op, Param):
                value = op.value
                unc = op.unc
            else:
                raise TypeError('operation type is not recognized')
            self.value = value
            self.unc = unc
        self.isuptodate = True
        return

    def _register_in_db(self, name):
        """Register a parameter in database."""
        if name in _dparams and 'obj' in _dparams[name]:
            self.logger.info('%s already registred, remove existing info',
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
    def _build_math_formula(op, par):
        strform = None
        par_strform = Param._get_strform(par)
        if par_strform != None:
            strform = '%s(%s)' % (op, par_strform)
        return strform

    @staticmethod
    def _build_str_formula(par1, op, par2):
        strform = None
        par1_strform = Param._get_strform(par1)
        par2_strform = Param._get_strform(par2)
        if par1_strform != None and par2_strform != None:
            if op != '+' and op != '-':
                if isinstance(par1, Param) and par1.name == None:
                    par1_strform = '(' + par1_strform + ')'
                if isinstance(par2, Param) and par2.name == None:
                    par2_strform = '(' + par2_strform + ')'
            strform = '%s %s %s' % (par1_strform, op, par2_strform)
        return strform

class PF(object):
    """Base class to define a Probability Function. 

       Probability Function is a generic name which includes both the
       probability mass function for discrete random variables and the
       probability density fucntion for continuous random variables.
       The function itself is defined in ``self.func``:

       * For a RAW PF, it is a pdf or a pmf of scipy.stats. 
       * For a DERIVED PF, it is list containing an operator and 
         pointers to other PFs.

       Attributes
       ----------
       name : str (optional)
           Function name
       func : scipy.stats.distributions.rv_generic, list
           Probability Density Function object.
       params : statspy.core.Param list
           List of shape parameters used to define the pf
       norm : Param
           Normalization parameter set to 1 by default. It can be different
           from 1 when the PF is fitted to data.
       isuptodate : bool
           Tells whether PF needs to be normalized or not
       options : dict
           Potential list of options
       pftype : PF.RAW or PF.DERIVED
           Tells whether a PF is a RAW PF or DERIVED from other PFs
       logger : logging.Logger
           message logging system

       Examples
       --------
       >>> import statspy as sp
       >>> pmf_n = sp.PF("pmf_n=poisson(n;mu)",mu=10.)
    """

    # Define the different probability function types
    (RAW,DERIVED) = (0,10)

    def __init__(self,*args,**kwargs):
        # Public class members
        self.name = None
        self.func = None
        self.params = []
        self.norm = Param(value=1., const=True)
        self.isuptodate = False
        self.options = kwargs.get('options', {})
        self.pftype = PF.RAW
        self.logger = logging.getLogger('statspy.core.PF')
        # Internal class members
        self._loc = None
        self._scale = None
        self._cache = None
        self._free_params = []
        self._pcov = None
        self._rvs = []
        self._spline_pf = None
        self._spline_cdf = None
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            self._check_kwargs_syntax(kwargs,foundArgs)
            if isinstance(self.func, scipy.stats.distributions.rv_generic):
                self.isuptodate = True
                self._rvs[0][1] = self.func.a # Lower bound
                self._rvs[0][2] = self.func.b # Upper bound
                if isinstance(self.func, scipy.stats.rv_discrete):
                    self._rvs[0][3] = RV.DISCRETE
                elif isinstance(self.func, scipy.stats.rv_continuous):
                    self._rvs[0][3] = RV.CONTINUOUS
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
            norm0 = [self.norm.value, other.norm.value]
            self_params  = self._get_add_norm_params()
            other_params = other._get_add_norm_params()
            add_params = self_params + other_params
            frac0 = 1./len(add_params)
            first = True
            for ipar,add_param in enumerate(add_params):
                if not ipar: continue
                if add_param.norm.partype == Param.RAW:
                    if add_param.norm.value >= 1.:
                        add_param.norm.value = frac0
                    add_param.norm.const = False
                    add_param.norm.bounds = [0., 1.]
                else:
                    add_param.norm = Param(value=frac0, bounds=[0., 1.],
                                           const=False)
                    if add_param.name != None:
                        add_param.name = 'norm_%s' % add_param.name
                if first:
                    add_params[0].norm = 1. - add_param.norm
                    first = False
                else:
                    add_params[0].norm -= add_param.norm
            if add_params[0].name != None:
                add_params[0].norm.name = 'norm_%s' % add_params[0].name
            for ele in [self_params, other_params]:
                if len(ele) < 2: continue
                first = True
                for ipar,add_param in enumerate(ele):
                    if ipar == 0: continue
                    if first:
                        ele.norm = ele[0].norm + add_param.norm
                        first = False
                    else:
                        ele.norm += add_param.norm
                if ele.name != None:
                    ele.norm.name = 'norm_%s' % ele.name
            new = PF(func=[operator.add, self, other])
            if norm0[0] != 1 or norm0[1] != 1:
                new.norm.value = norm0[0] + norm0[1]
        except:
            raise
        return new

    def __call__(self, *args, **kwargs):
        """Evaluate Probability Function in x

        Parameters
        ----------
        args : float, ndarray, optional, multiple values for multivariate pfs
            Random Variable(s) value(s)
        kwargs : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        value : float, ndarray
            Probability Function value(s) in x

        """
        # Check if self.func contains a pdf() or a pmf() method
        if self.pftype == PF.RAW:
            method_name = 'pdf'
            try:
                if isinstance(self.func, scipy.stats.rv_discrete):
                    method_name = 'pmf'
                check_method_exists(obj=self.func,name=method_name)
            except:
                raise
        # Get random variable value(s), mandatory
        rv_values = self._get_rv_values(*args, **kwargs)
        # Get shape parameters, optional
        param_values = self._get_param_values(**kwargs)
        # Compute pf value in x
        value = None
        #  - in case of a DERIVED PF, call an operator
        if (self.pftype == PF.DERIVED and isinstance(self.func, list) and
            len(self.func) == 1 and isinstance(self.func[0], PF)):
            value = self.func[0](*rv_values, **kwargs)
        elif self.pftype == PF.DERIVED:
            if not isinstance(self.func, list) or len(self.func) != 3:
                raise SyntaxError('DERIVED function is not recognized.')
            op = self.func[0] # operator
            if type(op) == str:
                redocache = not self.isuptodate or self._spline_pf == None
                for name,value in zip(self.rv_names(), rv_values):
                    if (np.amin(value) < np.amin(self._cache[name]) or
                        np.amax(value) > np.amax(self._cache[name])):
                        redocache = True
                if redocache: self._build_cache(*rv_values)
                value = self._spline_pf(*rv_values)
            else:
                vals = [0.,0.]
                for idx in [1,2]:
                    the_args = []
                    for irv,rv in enumerate(self._rvs):
                        if rv[0] in self.func[idx].rv_names():
                            the_args.append(rv_values[irv])
                    vals[idx-1] = self.func[idx](*the_args, **kwargs)
                value = op(vals[0],vals[1])
        else:
            #  - in case of a RAW PF, call directly the scipy function
            if method_name == 'pmf': 
                value = self.func.pmf(rv_values[0], *param_values)
            else:
                value = self.func.pdf(rv_values[0], *param_values)
        if self._scale != None and self._scale.value != 0.:
            value = value / self._scale.value
        return self.norm.value * value

    def __mul__(self, other):
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

    def __rmul__(self, scale):
        """Scale PF normalization value
        
        parameters
        ----------
        self : PF
        scale : float

        returns
        -------
        self : PF
            original PF with self.norm.value *= scale
        
        """
        try:
            self.norm.value *= scale
        except:
            raise
        return self

    def cdf(self, *args, **kwargs):
        """Compute the cumulative distribution function in x.

        Parameters
        ----------
        args : ndarray, tuple
            Random Variable(s) value(s)
        kwargs : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        value : float, ndarray
            Cumulative distribution function value(s) in x

        """
        # Check if self.func contains a cdf() method
        if self.pftype == PF.RAW:
            try:
                check_method_exists(obj=self.func,name='cdf')
            except:
                raise
        # Get random variable value(s), mandatory
        rv_values = self._get_rv_values(*args, **kwargs)
        # Get shape parameters, optional
        param_values = self._get_param_values(**kwargs)
        # Compute cdf value in x
        #  - in case of a DERIVED PF, call an operator
        if (self.pftype == PF.DERIVED and isinstance(self.func, list) and
            len(self.func) == 1 and isinstance(self.func[0], PF)):
            value = self.norm.value * self.func[0].cdf(*rv_values, **kwargs)
            return value
        elif self.pftype == PF.DERIVED:
            if not isinstance(self.func, list) or len(self.func) != 3:
                raise SyntaxError('DERIVED function is not recognized.')
            op = self.func[0]
            if type(op) == str:
                redocache = not self.isuptodate or self._spline_cdf == None
                for name,value in zip(self.rv_names(), rv_values):
                    if (np.amin(value) < np.amin(self._cache[name]) or
                        np.amax(value) > np.amax(self._cache[name])):
                        redocache = True
                if redocache: self._build_cache(*rv_values)
                value = self.norm.value * self._spline_cdf(*rv_values)
            else:
                vals = [0.,0.]
                for idx in [1,2]:
                    the_args = []
                    for irv,rv in enumerate(self._rvs):
                        if rv[0] in self.func[idx].rv_names():
                            the_args.append(rv_values[irv])
                    vals[idx-1] = self.func[idx].cdf(*the_args, **kwargs)
                value = self.norm.value * op(vals[0],vals[1])
            return value
        #  - in case of a RAW PF, call directly the scipy function
        return self.norm.value * self.func.cdf(rv_values[0], *param_values)

    def convolve(self, other, **kw):
        """Convolve two PFs which is same as adding two random variables.

        The scipy.signal package is used to perform the convolution with
        different options set by mode.

        parameters
        ----------
        self : PF
        other : PF
        kw : keywork arguments, dict
            May be:

            * options : str
                Way to perform the convolution. If mode is equal to:

                * ``fft`` then ``scipy.signal.fftconvolve`` is used
                * ``num`` then ``scipy.signal.convolve`` is used
                * ``rvs`` then random variates are used to generate the PF

        returns
        -------
        new : PF
            new PF which is the convolution of self and other

        Notes
        -----
        This method is working only for 1-dim pdf/pmf currently.

        """
        try:
            if not 'mode' in kw: kw['mode'] = 'fft'
            new = PF(func=['rvadd', self, other], options=kw)
        except:
            raise
        return new

    def corr(self):
        """Returns the correlation matrix of the free parameters.

        Returns
        -------
        corr : ndarray
            Correlation matrix

        """
        npars = len(self._free_params)
        if npars == 0: return None
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
        return corr

    def dF(self, x):
        """Compute the uncertainty on PF given the uncertainty on the shape
        and norm parameters.

        This method can be used to show an error band on your fitted PF.
        To compute the uncertainty on the PF, the error propagation formula is
        used::

            dF(x;th) = (F(x;th+dth) - F(x;th-dth))/2
            dF(x)^2 = dF(x;th)^T * corr(th,th') * dF(x;th')

        so keep in mind it is only an approximation.

        parameters
        ----------
        x : float, ndarray
            Random variable(s) value(s)

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
        mcorr = np.asmatrix(self.corr())
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
        if (not self.norm.const and self.norm.partype == Param.RAW and
            (not self.norm in self._free_params)):
            self._free_params.append(self.norm)
        if self.pftype == PF.DERIVED and type(self.func) == list:
            for ele in self.func:
                if not isinstance(ele, PF): continue
                if (ele.norm.partype == Param.RAW and not ele.norm.const and
                    (not ele.norm in self._free_params)):
                    self._free_params.append(ele.norm)
                elif ele.norm.partype == Param.DERIVED:
                    raw_params = ele.norm.get_raw_params()
                    for raw_par in raw_params:
                        if raw_par.const: continue
                        self._free_params.append(raw_par)
        # Get the list of shape parameters
        params = copy.copy(self.params)
        if self._loc != None: params.append(self._loc)
        if self._scale != None: params.append(self._scale)
        for par in params:
            if par.partype == Param.RAW and not par.const:
                self._free_params.append(par)
            elif par.partype == Param.DERIVED:
                raw_params = par.get_raw_params()
                for raw_par in raw_params:
                    if raw_par.const: continue
                    if raw_par in self._free_params: continue
                    self._free_params.append(raw_par)
        return self._free_params

    def kurtosis(self, **kw):
        """Estimate the kurtosis of the PF. See PF.mean() for the syntax."""
        kurtosis = self._estimate('k', scipy.stats.kurtosis, **kw)
        return kurtosis

    def leastsq_fit(self, xdata, ydata, ey=None, dx=None, cond=None, **kw):
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
        cond : boolean ndarray (optional)
            Boolean array telling if a bin should be used in the fit or not
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
             p0[ipar] = par.unbound_repr()
        # Compute weights
        if ey == None: ey = np.sqrt(ydata)
        ey[ey == 0] = 1.
        if cond == None:
            weight = 1./np.asarray(ey)
        else:
            weight = cond/np.asarray(ey)
        # Call the leastsq method
        if dx == None: dx = np.ones(xdata.shape)
        args = (xdata, ydata, weight, dx)
        res = scipy.optimize.leastsq(self._leastsq_function, p0,
                                     args=args, full_output=1, **kw)
        # Manage results
        (popt, self._pcov, infodict, errmsg, ier) = res
        self.logger.debug('Error message from leastsq: ' + str(ier) + 
                          " " + errmsg)
        if ier not in [1,2,3,4]:
            msg = "Optimal parameters not found: " + errmsg
            raise RuntimeError(msg)
        chi2min = (self._leastsq_function(popt, *args)**2).sum()
        if (len(ydata) > len(p0)) and self._pcov is not None:
            ndf = len(ydata)-len(p0)
            self._pcov = self._pcov * chi2min / ndf
            pvalue = scipy.stats.chi2.sf(chi2min, ndf)
            for ipar,par in enumerate(self._free_params):
                unc = 0.
                if (self._pcov)[ipar][ipar] >= 0.:
                    unc = math.sqrt((self._pcov)[ipar][ipar])
                val2, unc2 = par.unbound_to_bound(popt[ipar], unc)
                par._pcov = []
                for jpar, par2 in enumerate(self._free_params):
                    if unc == 0.: continue
                    self._pcov[ipar][jpar] *= (unc2 / unc)
                    self._pcov[jpar][ipar] *= (unc2 / unc)
                    if jpar != ipar:
                        par._pcov.append([par2, self._pcov[jpar][ipar]])
        else:
            self._pcov = np.inf
            pvalue = np.inf
        return self._free_params, self._pcov, chi2min, pvalue

    def loc(self, loc):
        """Derive a PF from another PF via a location parameter.
        
        parameters
        ----------
        self : PF
        loc : Param, value

        returns
        -------
        new : PF
            new PF with x -> x + loc
        
        """
        try:
            new = PF(func=[self])
            if isinstance(loc, Param):
                new._loc = loc
            else:
                new._loc = Param(value=loc)
        except:
            raise
        return new

    def logpf(self, *args, **kwargs):
        """Compute the logarithm of the PF in x.

        Parameters
        ----------
        args : ndarray, tuple
            Random Variable(s) value(s)
        kwargs : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        value : float, ndarray
            Logarithm of the Probability Function value(s) in x

        """
        # Check if self.func contains a logpdf() or a logpmf() method
        if self.pftype == PF.RAW:
            method_name = 'logpdf'
            try:
                if isinstance(self.func, scipy.stats.rv_discrete):
                    method_name = 'logpmf'
                check_method_exists(obj=self.func,name=method_name)
            except:
                raise
        # Get random variable value(s), mandatory
        rv_values = self._get_rv_values(*args, **kwargs)
        # Get shape parameters, optional
        param_values = self._get_param_values(**kwargs)
        # Compute logpf value in x
        value = None
        #  - in case of a DERIVED PF, call an operator
        if (self.pftype == PF.DERIVED and isinstance(self.func, list) and
            len(self.func) == 1 and isinstance(self.func[0], PF)):
            value = self.func[0].logpf(*rv_values, **kwargs)
            if self._scale != None and self._scale.value != 0.:
                value = value - self._scale.value
        if self.pftype == PF.DERIVED:
            if not isinstance(self.func, list) or len(self.func) != 3:
                raise SyntaxError('DERIVED function is not recognized.')
            op = self.func[0]
            if op == operator.add:
                value = np.log(self(*args, **kwargs))
            elif op == operator.mul:
                vals = [0.,0.]
                for idx in [1,2]:
                    the_args = []
                    for irv,rv in enumerate(self._rvs):
                        if rv[0] in self.func[idx].rv_names():
                            the_args.append(rv_values[irv])
                    vals[idx-1] = self.func[idx].logpf(*the_args, **kwargs)
                value = operator.add(vals[0],vals[1])
                if self._scale != None and self._scale.value != 0.:
                    value = value - self._scale.value
            else:
                raise NotImplementedError('Not yet possible...')
        else:
            #  - in case of a RAW PF, call directly the scipy function
            if method_name == 'logpmf':
                value = self.func.logpmf(rv_values[0], *param_values)
            else:
                value = self.func.logpdf(rv_values[0], *param_values)
            if self._scale != None and self._scale.value != 0.:
                value = value - self._scale.value
        return self.norm.value + value 

    def maxlikelihood_fit(self, data, **kw):
        """Fit the PF to data using the maximum likelihood estimator method.

        The fitting part is performed using the ``scipy.optimize.minimize`` 
        function. If keyword argument ``method`` is not specified,
        the BFGS algorithm is used.
        When calling this method, all PF parameters are minimized except
        the one which are set as 'const' before calling the method.

        Parameters
        ----------
        data : ndarray, tuple
            Data used in the computation of the (log-)likelihood function
        kw : keyword arguments (optional)
            Keyword arguments passed to the scipy.optimize.minimize method.

        Returns
        -------
        free_params : statspy.core.Param list
            List of the free parameters used during the fit. Their 'value'
            arguments are extracted from the minimization process. If the
            Hessian is provided by the algorithm, the 'unc' argument is also
            updated.
        nllfmin : float
            Minimal value of the negative log-likelihood function

        """
        # Define parameters which should be minimized and set initial values
        self.get_list_free_params()
        p0 = np.ones(len(self._free_params))
        for ipar,par in enumerate(self._free_params):
             p0[ipar] = par.unbound_repr()
        # Call the minimizer (BFGS method by default)
        kwargs = {'method':'Nelder-Mead'}
        for name in ['method', 'jac', 'hess', 'hessp', 'bounds',
                     'constraints', 'tol', 'callback', 'options']:
            if name in kw: kwargs[name] = kw[name]
        result = scipy.optimize.minimize(self._nllf, p0, 
                                         args=(data, ), **kwargs)
        # Manage results
        if hasattr(result, 'hess'):
            self._pcov = result.hess # which is in fact the inverse hessian
        else:
            self._pcov = None
        for ipar,par in enumerate(self._free_params):
            val = result.x[ipar]
            unc = 0.
            if self._pcov != None and (self._pcov)[ipar][ipar] >= 0.:
                unc = math.sqrt((self._pcov)[ipar][ipar])
            val2, unc2 = par.unbound_to_bound(val, unc)
            par._pcov = []
            for jpar, par2 in enumerate(self._free_params):
                if unc == 0.: continue
                self._pcov[ipar][jpar] *= (unc2 / unc)
                self._pcov[jpar][ipar] *= (unc2 / unc)
                if jpar != ipar:
                    par._pcov.append([par2, self._pcov[jpar][ipar]])
        nllfmin = self.nllf(data)
        return self._free_params, nllfmin

    def mean(self, **kw):
        """Estimate the mean of the PF.

        .. warning::

           * In the case of a RAW PF from ``scipy.stats``, it calls the ``stats`` 
             method and therefore returns the expected value.
           * In the case of a DERIVED PF, it returns an estimate derived from 
             random variates and using the ``numpy.mean`` function.

        Parameters
        ----------
        kw : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        mean : ndarray

        """
        mean = self._estimate('m', np.mean, **kw)
        return mean

    def nllf(self, data, **kw):
        """Evaluate the negative log-likelihood function::

            nllf = -sum_i(log(pf(x_i;params)))

        ``sum`` runs over the x-variates defined in data array.

        Parameters
        ----------
        data : ndarray, tuple
            x - variates used in the computation of the likelihood 
        kw : keyword arguments (optional)
            Specify any Parameter name of the considered PF

        Returns
        -------
        nllf : float
            Negative log-likelihood function

        """
        # Update values of non-const PF parameters if specified in **kw
        for par in self._free_params:
            if par.name in kw:
                par.value = kw[par.name]
        # Compute nllf
        if isinstance(data, tuple):
            nllf = -1 * np.sum(self.logpf(*data))
        else:
            nllf = -1 * np.sum(self.logpf(data))
        return nllf

    def pllr(self, data, **kw):
        """Evaluate the profile log-likelihood ratio ( * -2 )

        The profile likelihood ratio is defined by::

            l = L(x|theta_r,\hat{\hat{theta_s}}) / L(x|\hat{theta_r},\hat{theta_s})

        The profile log-likelihood ratio is then::

            q = -2 * log(l)

        Where

        * L is the Likelihood function (computed via data)
        * theta_r is the list of parameters of interest (Param.poi = True)
        * theta_s is the list of nuisance parameters
        * hat or double hat refers to the unconditional and conditional
          maximum likelilood estimates of the parameters respectively.

        pllr is used as a test statistics for problems with numerous 
        nuisance parameters. Asymptotically, the pllr PF is described by a
        chi2 distribution (Wilks theorem).
        Further information on the likelihood ratio can be found in Chapter 
        22 of "Kendall's Advanced Theory of Statistics, Volume 2A".

        Parameters
        ----------
        data : ndarray, tuple
            x - variates used in the computation of the likelihood 
        kw : keyword arguments (optional)
            Specify any Parameter of interest name of the considered PF,
            or any option used by the method maxlikelihood_fit.

            uncond_nllf : float
                unconditional minimal negative log-likelihood function value

        Returns
        -------
        pllr : float
            Profile log-likelihood ratio times -2

        """
        # Find the free parameters and the parameters of interest 
        # Update values of non-const PF parameters if specified in **kw
        self.get_list_free_params()
        lpois = []
        for par in self._free_params:
            if not par.poi: continue
            lpois.append(par)
            if par.name in kw:
                par.value = kw[par.name]
        # Compute the conditional nllf (pois are fixed)
        for idx,poi in enumerate(lpois):
            poi.const = True
        cond_params, cond_nllf = self.maxlikelihood_fit(data, **kw)
        # Compute the unconditional nllf (pois are treated as free parameters)
        # if not provided
        for idx,poi in enumerate(lpois):
            poi.const = False
        uncond_nllf = kw.get('uncond_nllf', None)
        if uncond_nllf == None:
            uncond_params, uncond_nllf = self.maxlikelihood_fit(data, **kw)
        # Evaluate the pllr
        pllr = 2. * (cond_nllf - uncond_nllf)
        return pllr

    def pvalue(self, *args, **kwargs):
        """Compute the pvalue in xobs.

        The p-value is the probability of observing at least xobs 
        Pr(x >= xobs).

        Parameters
        ----------
        args : ndarray, tuple
            Random Variable(s) value(s)
        kwargs : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        pvalue : float, ndarray
            p-value(s) in x

        """
        try:
            pvalue = self.sf(*args, **kwargs)
            # Since p-value is probability of observing at least xobs,
            # add the probability in xobs when discrete.
            all_discrete = True
            for rv in self._rvs:
                if rv[3] == RV.DISCRETE: continue
                all_discrete = False
                break
            if all_discrete:
                pvalue = pvalue + self(*args, **kwargs)
        except:
            raise
        return pvalue

    def rvadd(self, other, **kw):
        """Operation equivalent to adding two random variables."""
        try:
            new = self.convolve(other, **kw)
        except:
            raise
        return new

    def rvdiv(self, other, **kw):
        """Operation equivalent to dividing two random variables."""
        try:
            new = PF(func=['rvdiv', self, other], options=kw)
        except:
            raise
        return new

    def rvmul(self, other, **kw):
        """Operation equivalent to multiplying two random variables."""
        try:
            new = PF(func=['rvmul', self, other], options=kw)
        except:
            raise
        return new

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
            auto_loc = False
            if isinstance(self.func, list):
                if len(self.func) == 3:
                    auto_loc = True
                    if self.func[0] == operator.add:
                        data1 = self.func[1].rvs(**kwargs)
                        data2 = self.func[2].rvs(**kwargs)
                        data3 = scipy.stats.uniform.rvs(size=len(data1))
                        cond = (data3 < self.func[1].norm.value)
                        data = cond * data1 + (1 - cond) * data2
                    elif self.func[0] == 'rvadd':
                        data1 = self.func[1].rvs(**kwargs)
                        data2 = self.func[2].rvs(**kwargs)
                        data = data1 + data2
                    elif self.func[0] == 'rvsub':
                        data1 = self.func[1].rvs(**kwargs)
                        data2 = self.func[2].rvs(**kwargs)
                        data = data1 - data2
                    elif self.func[0] == 'rvmul':
                        data1 = self.func[1].rvs(**kwargs)
                        data2 = self.func[2].rvs(**kwargs)
                        data = data1 * data2
                    elif self.func[0] == 'rvdiv':
                        data1 = self.func[1].rvs(**kwargs)
                        data2 = self.func[2].rvs(**kwargs)
                        data = data1 / data2
                    else:
                        raise NotImplementedError('Not yet implemented...')
                elif len(self.func) == 1 and isinstance(self.func[0], PF):
                    data = self.func[0].rvs(**kwargs)
                else:
                    raise NotImplementedError('Not yet implemented...')
            else:
                method_name = "rvs"
                check_method_exists(obj=self.func,name=method_name)
                shape_params = []
                for param in self.params:
                    if param.name in kwargs:
                        param.value = kwargs[param.name]
                    shape_params.append(param.value)
                data = self.func.rvs(*shape_params, **kwargs)
            if self._scale != None and self._scale.value != 0.:
                data = data * self._scale.value
            if self._loc != None and not auto_loc:
                data = data + self._loc.value
        except:
            raise
        return data

    def rvsub(self, other, **kw):
        """Operation equivalent to subtracting two random variables."""
        try:
            if not 'mode' in kw: kw['mode'] = 'fft'
            new = PF(func=['rvsub', self, other], options=kw)
        except:
            raise
        return new

    def rv_names(self):
        """Return a list of RV names."""
        rv_names = [ rv[0] for rv in self._rvs ]
        return rv_names

    def scale(self, scale):
        """Derive a PF from another PF via a scale parameter.
        
        parameters
        ----------
        self : PF
        scale : Param, value

        returns
        -------
        new : PF
            new PF with x -> x * scale
        
        """
        try:
            new = PF(func=[self])
            if isinstance(scale, Param):
                new._scale = scale
            else:
                new._scale = Param(value=scale)
        except:
            raise
        return new

    def sf(self, *args, **kwargs):
        """Compute the survival function (1 - cdf) in x.

        Parameters
        ----------
        args : ndarray, tuple
            Random Variable(s) value(s)
        kwargs : keywork arguments, optional
            Shape parameters values

        Returns
        -------
        value : float, ndarray
            Survival function value(s) in x

        """
        # Check if self.func contains an sf() method
        if self.pftype == PF.RAW:
            try:
                check_method_exists(obj=self.func,name='sf')
            except:
                raise
        # Get random variable value(s), mandatory
        rv_values = self._get_rv_values(*args, **kwargs)
        # Get shape parameters, optional
        param_values = self._get_param_values(**kwargs)
        # Compute sf value in x
        #  - in case of a DERIVED PF, call an operator
        if (self.pftype == PF.DERIVED and isinstance(self.func, list) and
            len(self.func) == 1 and isinstance(self.func[0], PF)):
            value = self.norm.value * self.func[0].sf(*rv_values, **kwargs)
            return value
        elif self.pftype == PF.DERIVED:
            if not isinstance(self.func, list) or len(self.func) != 3:
                raise SyntaxError('DERIVED function is not recognized.')
            op = self.func[0]
            if type(op) == str:
                redocache = not self.isuptodate or self._spline_cdf == None
                for name,value in zip(self.rv_names(), rv_values):
                    if (np.amin(value) < np.amin(self._cache[name]) or
                        np.amax(value) > np.amax(self._cache[name])):
                        redocache = True
                if redocache: self._build_cache(*rv_values)
                value = 1 - self.norm.value * self._spline_cdf(*rv_values)
            else:
                vals = [0.,0.]
                for idx in [1,2]:
                    the_args = []
                    for irv,rv in enumerate(self._rvs):
                        if rv[0] in self.func[idx].rv_names():
                            the_args.append(rv_values[irv])
                    vals[idx-1] = self.func[idx].sf(*the_args, **kwargs)
                value = self.norm.value * op(vals[0],vals[1])
            return value
        #  - in case of a RAW PF, call directly the scipy function
        return self.norm.value * self.func.sf(rv_values[0], *param_values)

    def skew(self, **kw):
        """Estimate the skewness of the PF. See PF.mean() for the syntax."""
        skew = self._estimate('s', scipy.stats.skew, **kw)
        return skew

    def std(self, **kw):
        """Estimate the standard deviation of the PF.
        See PF.mean() for the syntax."""
        var = self._estimate('v', np.var, **kw)
        return np.sqrt(var)

    def var(self, **kw):
        """Estimate the variance of the PF. See PF.mean() for the syntax."""
        var = self._estimate('v', np.var, **kw)
        return var

    def _build_cache(self, rv_values=None):
        if not isinstance(self.func, list):
            raise SyntaxError('_build_cache, self.func is not a list.')
        if len(self.func) < 3:
            raise SyntaxError('self.func has a size lower than 3.')
        if type(self.func[0]) != str:
            raise SyntaxError('First element of self.func should be a str.')
        if len(self.func[1]._rvs) != len(self.func[2]._rvs):
            raise SyntaxError('self and other should have the same number of rvs.')
        for rv1,rv2 in zip(self.func[1]._rvs,self.func[2]._rvs):
            if rv1[3] == rv2[3]: continue
            raise SyntaxError('%s and %s should be of the same type.' % (rv1[0], rv2[0]))
        (getattr(self, ('_'+self.func[0])))(rv_values=rv_values)
        self.isuptodate = True
        return

    def _build_cache_rvs(self):
        """Generate random variates and build an histogram"""
        size0 = max(10**6, 100000 * 10**(len(self._rvs) - 1))
        size = self.options.get('size', size0)
        data = self.rvs(size=size)
        if self._rvs[0][3] == RV.DISCRETE:
            xmin = data.min() - 0.5
            xmax = data.max() + 0.5
            nbins = xmax - xmin
            pf, bins = np.histogram(data, bins=nbins, range=(xmin, xmax))
        else:
            pf, bins = np.histogram(data, bins=100)
        rv_name = self._rvs[0][0]
        dtype = [(rv_name,float),('pf',float),('cdf',float)]
        self._cache = np.recarray(pf.shape, dtype=dtype)
        self._cache[rv_name] = 0.5*(bins[1:] + bins[:-1]) # Bin centers
        self._cache.pf = pf                               # Bin contents
        if self._rvs[0][3] == RV.CONTINUOUS:
            dx = bins[1:] - bins[:-1]                     # Bin widths
            integral = (self._cache.pf * dx).sum()
            self._cache.cdf = np.cumsum(self._cache.pf * dx)
        else:
            integral = (self._cache.pf).sum()
            self._cache.cdf = np.cumsum(self._cache.pf)
        self.norm.value = 1. / integral

    def _build_spline(self):
        if self._cache == None:
            raise SyntaxError('cache is empty, cannot build the spline')
        if len(self._rvs) > 2: return
        if len(self._rvs) == 1:
            name = self._rvs[0][0]
            method = self.options.get('spline',
                                      scipy.interpolate.UnivariateSpline)
            self._spline_pf = method(self._cache[name], self._cache.pf)
            self._spline_cdf = method(self._cache[name], self._cache.cdf)
        elif len(self._rvs) == 2:
            method = self.options.get('spline',
                                      scipy.interpolate.RectBivariateSpline)
            name1 = self._rvs[0][0]
            name2 = self._rvs[1][0]
            self._spline_pf = method(self._cache[name1], self._cache[name2],
                                     self._cache.pf)
            self._spline_cdf = method(self._cache[name1], self._cache[name2],
                                      self._cache.cdf)
        return

    def _check_args_syntax(self, args):
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
            if not rv_name in self.rv_names():
                self._rvs.append([rv_name,-np.inf,np.inf,RV.UNKNOWN])
        lpars = []
        if parNames != None:
            for par_name in parNames.split(','):
                lpars.append(par_name.strip().lstrip())
        self._declare(func_name,lpars)
        return True

    def _check_kwargs_syntax(self, kwargs, foundArgs):
        if not len(kwargs): return False
        if 'name' in kwargs:
            if self.name != None:
                raise SyntaxError("self.name is already set to %s" % self.name)
            self.name = kwargs['name']
            self.norm.name = 'norm_%s' % self.name
            for par in self.params:
                if par.name == None: continue
                if not par.name in _dparams: continue
                _dparams[par.name]['pfs'].append(self.name)
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
                    for rv in ele._rvs:
                        if (type(self.func[0]) == str and
                            self.func[0].startswith('rv')):
                            # For operations on RVs, the number of RVs is 
                            # equal to the number of RVs for func[1] or 
                            # func[2]
                            if len(self._rvs) < len(self.func[1]._rvs):
                                self._rvs.append(rv)
                        elif not rv[0] in self.rv_names():
                            self._rvs.append(rv)
                if len(self.func) > 1 and type(self.func[0]) == str:
                    # Tricky operators like convolution requiring a cache
                    self._build_cache()
            else:
                self.pftype = PF.RAW
        if 'param' in kwargs:
            if len(self.params) != 0: 
                raise SyntaxError("self.params are already declared.")
            if not isinstance(kwargs['params'], list):
                raise SyntaxError("params should be a list.")
            self.params = kwargs['params']
            for par in self.params:
                if par.name == None or self.name == None: continue
                if not par.name in _dparams: continue
                _dparams[par.name]['pfs'].append(self.name)
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
        return True

    def _declare(self, func_name, lpars):
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

    def _estimate(self, stats, method, **kw):
        if self.pftype == PF.RAW:
            method_name = 'stats'
            try:
                check_method_exists(obj=self.func,name=method_name)
                if not stats in ['m','v','s','k']:
                    raise SyntaxError("stats str should be among ['mvsk']")
            except:
                raise
            # Get shape parameters, optional
            param_values = self._get_param_values(**kw)
            estimate = self.func.stats(*param_values, moments=stats)
        else:
            if not 'size' in kw: kw['size'] = 10000
            data = self.rvs(**kw)
            estimate = method(data)
        return estimate

    def _get_add_norm_params(self):
        norm_params = []
        if self.pftype == PF.DERIVED and self.func[0] == operator.add:
            for ele in self.func:
                if not isinstance(ele, PF): continue
                raw_norm_params += ele._get_add_norm_params()
        elif self.norm.partype == PF.RAW:
            norm_params.append(self)
        return norm_params

    def _get_bounds(self):
        bounds = []
        for rv in self._rvs:
            a = rv[1]
            b = rv[2]
            mean = self.mean()
            if a == -np.inf or b == np.inf:
                nsigma = 10
                std  = self.std()
                if a == -np.inf:
                    c = mean - nsigma * std
                else:
                    c = a
                if b == np.inf:
                    d = mean + nsigma * std
                else:
                    d = b
            else:
                c = a
                d = b
            if rv[3] == RV.DISCRETE: mean = round(mean)
            bounds.append([a, b, c, d, mean])
        return bounds

    def _get_param_values(self, **kwargs):
        param_values = [0.] * len(self.params)
        for ipar,param in enumerate(self.params):
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
            param_values[ipar] = param.value
        return param_values

    def _get_rv_values(self, *args, **kwargs):
        rv_values = []
        if len(args):
            for arg in args:
                rv_values.append(arg)
        else:
            rv_values = []
            for rv_name in self.rv_names():
                if rv_name in kwargs: rv_values.append(kwargs[rv_name])
        if self.pftype == PF.RAW and len(rv_values) != len(self._rvs):
            raise SyntaxError('Provide %s input arguments for rvs' % 
                              len(self._rvs))
        if self._loc != None and len(rv_values) == 1:
            rv_values[0] = rv_values[0] - self._loc.value
        if (self._scale != None and len(rv_values) == 1 and 
            self._scale.value != 0.):
            rv_values[0] = rv_values[0] / self._scale.value
        return rv_values

    def _leastsq_function(self, params, xdata, ydata, weight, dx):
        """Function used by scipy.optimize.leastsq"""
        # Update values of non-const PF parameters
        for ipar,par in enumerate(self._free_params):
            par.unbound_to_bound(params[ipar])
        # Return delta = (PF(x) - y)/sigma
        delta = weight * (self(xdata) * dx - ydata)
        return delta

    def _nllf(self, params, data):
        """Evaluate the negative log-likelihood function using unbound 
        parameters"""
        # Update values of non-const PF parameters
        for ipar,par in enumerate(self._free_params):
            par.unbound_to_bound(params[ipar])
        # Compute nllf
        if isinstance(data, tuple):
            nllf = -1 * np.sum(self.logpf(*data))
        else:
            nllf = -1 * np.sum(self.logpf(data))
        return nllf

    def _rvadd(self, rv_values=None):
        """Convolve numerically two PFs and store the result in _cache."""
        mode = self.options.get('mode', 'fft')
        if mode == 'num' or mode == 'fft':
            # Define the rv bounds
            rv1_bounds = self.func[1]._get_bounds()
            rv2_bounds = self.func[2]._get_bounds()
            bounds = []
            npts = self.options.get('size', 1001)
            for rv1,rv2 in zip(rv1_bounds,rv2_bounds):
                mean = None
                if self.func[0] == 'rvadd':
                    a = rv1[0] + rv2[0]
                    b = rv1[1] + rv2[1]
                    c = rv1[2] + rv2[2]
                    d = rv1[3] + rv2[3]
                    if a == -np.inf and b == np.inf:
                        if self._loc == None: self._loc = Param(value=1.)
                        self._loc.value = rv1[4] + rv2[4]
                        c = c - self._loc.value
                        d = d - self._loc.value
                else:
                    a = rv1[0] - rv2[1]
                    b = rv1[1] - rv2[0]
                    c = rv1[2] - rv2[3]
                    d = rv1[3] - rv2[2]
                    if self._loc == None: self._loc = Param(value=1.)
                    self._loc.value = rv1[4] - rv2[4]
                    wdt = (d - c) / 2
                    c = -wdt
                    d = wdt
                bounds.append([a, b, c, d])
            # TODO, algo is problematic when > 1 rv
            for rv,bound in zip(self._rvs,bounds):
                if rv[3] == RV.CONTINUOUS: continue
                bound[2] = a = round(bound[2])
                bound[3] = b = round(bound[3])
                if b - a + 1 > npts:
                    step = round((b - a)/float(npts - 1))
                    npts = (b - a)/step + 1
                else:
                    npts = b - a + 1
            # Arrays of axis coordinates
            dtype = []
            for rv in self._rvs:
                if rv[3] == RV.CONTINUOUS:
                    dtype.append((rv[0],float))
                else:
                    dtype.append((rv[0],int))
            dtype.append(('pf',float))
            dtype.append(('cdf',float))
            self._cache = np.recarray((npts,),dtype=dtype)
            args1 = []
            args2 = []
            bin_area = 1.
            step = None
            for rv,bound in zip(self._rvs,bounds):
                rv[1] = bound[0]
                rv[2] = bound[1]
                self._cache[rv[0]], step = np.linspace(bound[2], bound[3],
                                                       num=npts, retstep=True)
                bin_area *= step
                if self._loc != None:
                    args1.append(self._cache[rv[0]] + rv1_bounds[0][4])
                    args2.append(self._cache[rv[0]] + rv2_bounds[0][4])
                else:
                    args1.append(self._cache[rv[0]])
                    args2.append(self._cache[rv[0]])
                self.logger.debug('%s bounds set to %s' % (rv[0],bound))
            # Evaluate pf values for both input pf
            pf1 = self.func[1](*args1)
            pf2 = self.func[2](*args2)
            if self.func[0] == 'rvsub': pf2 = pf2[::-1]
            # Perform the convolution
            if mode == 'num':
                pf = scipy.signal.convolve(pf1, pf2, mode='full')
            else:
                pf = scipy.signal.fftconvolve(pf1, pf2, mode='full')
            if self.func[0] == 'rvadd' and self._rvs[0][1] != -np.inf:
                start = 0
            elif self.func[0] == 'rvsub' and 0 in self._cache[rv[0]]:
                start = npts - 1 - np.where(self._cache[rv[0]] == 0)[0][0]
            else:
                start = (npts - 2) / 2
            self._cache.pf = pf[start:start+npts]
            # Cumulative distribution function
            self._cache.cdf = np.cumsum(self._cache.pf * bin_area)
            # Normalization factor
            integral = np.sum(self._cache.pf * bin_area)
            self.norm.value = 1. / integral
            # Interpolation setup
            if not 'spline' in self.options:
                if len(self._rvs) == 1:
                    self.options['spline'] = scipy.interpolate.interp1d
                elif len(self._rvs) == 2:
                    self.options['spline'] = scipy.interpolate.interp2d
                else:
                    self.options['spline'] = scipy.interpolate.griddata
        elif len(self._rvs) == 1:
            # Generate random variated and build an histogram
            self._build_cache_rvs()
        else:
            raise NotImplementedError('Sorry, rvadd operation not valid...')
        # Interpolate
        self._build_spline()
        return

    def _rvdiv(self, rv_values=None):
        self._build_cache_rvs()
        # Interpolate
        self._build_spline()
        return

    def _rvmul(self, rv_values=None):
        self._build_cache_rvs()
        # Interpolate
        self._build_spline()
        return

    def _rvsub(self, rv_values=None):
        self._rvadd(rv_values=rv_values)
        return

class RV(object):
    """Base class to define a Random Variable. 

       Attributes
       ----------
       name : str
           Random Variable name
       pf : statspy.core.PF
           Probability Function object associated to a Random Variable
       rvtype : RV.CONTINUOUS or RV.DISCRETE
           Random Variable type
       logger : logging.Logger
           message logging system

       Examples
       --------
       >>> import statspy as sp 
       >>> X = sp.RV("norm(x|mu=10,sigma=2)")
    """
    # Define the different random value types
    (UNKNOWN,CONTINUOUS,DISCRETE) = (0,10,100)

    def __init__(self, *args, **kwargs):
        self.name = ""
        self.pf = None
        self.rvtype = RV.UNKNOWN
        self.logger = logging.getLogger('statspy.core.RV')
        try:
            self.logger.debug('args = %s, kwargs = %s',args, kwargs)
            foundArgs = self._check_args_syntax(args, kwargs)
            self._check_kwargs_syntax(kwargs, foundArgs)
        except:
            raise

    def __add__(self, other):
        """Add two random variables or a random variable by a parameter.

        The associated PF of the sum of the two random variables is
        a convolution of the two PFs.
        Caveat: this method assumes *independent* random variables.
        
        Parameters
        ----------
        self : RV
        other : RV, Param, value

        Returns
        -------
        new : RV
             new RV which is the sum of self and other
        
        """
        try:
            if isinstance(other, RV):
                new = RV(pf=self.pf.rvadd(other.pf))
            else:
                new = RV(pf=self.pf.loc(other))
        except:
            raise
        return new

    def __call__(self, **kwargs):
        """Get random variates

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
        >>> X = sp.RV("norm(x;mu=20,sigma=5)")
        >>> x = X(size=1000)

        """
        try:
            if not isinstance(self.pf, PF):
                raise SyntaxError('No PF associated to this random variable.')
            data = self.pf.rvs(**kwargs)
        except:
            raise
        return data

    def __div__(self, other):
        """Divide two random variables or a random variable by a parameter.

        Caveat: this method assumes *independent* random variables.
        
        Parameters
        ----------
        self : RV
        other : RV, Param, value

        Returns
        -------
        new : RV
             new RV which is the ratio of self and other
        
        """
        try:
            if isinstance(other, RV):
                new = RV(pf=self.pf.rvdiv(other.pf))
            else:
                new = RV(pf=self.pf.scale((1./other)))
        except:
            raise
        return new

    def __mul__(self, other):
        """Multiply two random variables or a random variable by a parameter.

        Caveat: this method assumes *independent* random variables.
        
        Parameters
        ----------
        self : RV
        other : RV, Param, value

        Returns
        -------
        new : RV
             new RV which is the product of self and other
        
        """
        try:
            if isinstance(other, RV):
                new = RV(pf=self.pf.rvmul(other.pf))
            else:
                new = RV(pf=self.pf.scale(other))
        except:
            raise
        return new

    def __radd__(self, other):
        """Add a random variable and a parameter.

        Parameters
        ----------
        self : RV
        other : Param, value

        Returns
        -------
        new : RV
             new RV which is the sum of self and other
        
        """
        try:
            new = RV(pf=self.pf.loc(other))
        except:
            raise
        return new

    def __rmul__(self, other):
        """Multiply a parameter by a random variable.

        Parameters
        ----------
        self : RV
        other : Param, value

        Returns
        -------
        new : RV
             new RV which is the product of self and other
        
        """
        try:
            new = RV(pf=self.pf.scale(other))
        except:
            raise
        return new

    def __sub__(self, other):
        """Subtract two random variables or a random variable by a parameter.

        Caveat: this method assumes *independent* random variables.
        
        Parameters
        ----------
        self : RV
        other : RV, Param, value

        Returns
        -------
        new : RV
             new RV which is the difference of self and other
        
        """
        try:
            if isinstance(other, RV):
                new = RV(pf=self.pf.rvsub(other.pf))
            else:
                new = RV(pf=self.pf.loc(((-1.)*other)))
        except:
            raise
        return new

    def _check_args_syntax(self, args, kwargs):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to PF without a keyword, it must be a string.")
        self.pf = PF(*args, **kwargs)
        if len(self.pf._rvs) != 1:
            raise SyntaxError("One and only one RV should be found in %s." % args)
        self.name = self.pf._rvs[0][0]
        self.rvtype = self.pf._rvs[0][3]
        if self.pf.name == None:
            if self.rvtype == RV.CONTINUOUS:
                self.pf.name = 'pdf_%s' % self.name
            else:
                self.pf.name = 'pmf_%s' % self.name
        return True

    def _check_kwargs_syntax(self, kwargs, foundArgs):
        if not len(kwargs): return False
        if not foundArgs and not 'pf' in kwargs:
            raise SyntaxError("You cannot declare a Random Variable without specifying a pf.")
        if 'name' in kwargs:
            if self.name != None:
                raise SyntaxError("self.name is already set to %s" % self.name)
            self.name = kwargs['name']
        if 'pf' in kwargs: self.pf = kwargs['pf']
        if 'rvtype' in kwargs: self.rvtype = kwargs['rvtype']
        return True

def check_method_exists(obj=None, name=""):
    if obj == None:
        raise StandardError('Object is not defined, check syntax.')
    if (not hasattr(obj,name) or 
        not callable(getattr(obj,name))):
        raise StandardError('No %s() method found.' % name)
    return True

def exp(x):
    """Compute the exponential of a Parameter.

        Parameters
        ----------
        x : Param
            Input parameter
    
        Returns
        -------
        y : Param
            Exponential of x
    
        Examples
        --------
        >>> import statspy as sp
        >>> x = sp.Param("x = 4 +- 1")
        >>> y = sp.exp(x)
    """
    try:
        if isinstance(x, Param):
            y = Param(formula=[[math.exp, x]],
                      derivative=lambda z : math.exp(z),
                      strform=Param._build_math_formula('exp', x))
        else:
            raise SyntaxError('exp function only applies to Parameters.')
    except:
        raise
    return y

def get_obj(obj_name):
    """Returns a Param, PF or RV object.

    Look in the different dictionaries if an object named obj_name exists
    and returns it.

    Parameters
    ----------
    obj_name : str
        Object name used to define a Param, PF or RV

    Returns
    -------
    new : Param, PF, RV
         Object if found in the dictionaries

    Examples
    --------
    >>> import statspy as sp
    >>> mypmf = sp.PF('mypmf=poisson(n;lbda=5)')
    >>> lbda = sp.get_obj('lbda')
    >>> lbda.label = '\\\\lambda'

    """
    obj = None
    for obj_dict in (_drvs, _dpfs, _dparams):
        if not obj_name in obj_dict: continue
        if obj != None:
            logger.warning('%s is found multiple times !' % obj_name)
        logger.debug('%s has been found in %s' % (obj_name, obj_dict.keys()))
        obj = obj_dict[obj_name]['obj']
    return obj

def log(x):
    """Compute the logarithm of a Parameter.

        Parameters
        ----------
        x : Param
            Input parameter
    
        Returns
        -------
        y : Param
            Logarithm of x
    
        Examples
        --------
        >>> import statspy as sp
        >>> x = sp.Param("x = 4 +- 1")
        >>> y = sp.log(x)
    """
    try:
        if isinstance(x, Param):
            y = Param(formula=[[math.log, x]],
                      derivative=lambda z : 1./z,
                      strform=Param._build_math_formula('log', x))
        else:
            raise SyntaxError('log function only applies to Parameters.')
    except:
        raise
    return y

def sqrt(x):
    """Compute the square root of a Parameter.

        Parameters
        ----------
        x : Param
            Input parameter
    
        Returns
        -------
        y : Param
            Square root of x
    
        Examples
        --------
        >>> import statspy as sp
        >>> x = sp.Param("x = 4 +- 1")
        >>> y = sp.sqrt(x)
    """
    try:
        if isinstance(x, Param):
            y = Param(formula=[[math.sqrt, x]],
                      derivative=lambda z : 1./2./math.sqrt(z),
                      strform=Param._build_math_formula('sqrt', x))
        else:
            raise SyntaxError('sqrt function only applies to Parameters.')
    except:
        raise
    return y

def update_status(obj_dict):
    """Check the list of objects depending on this parameter and update
    their isuptodate status to False.
    """
    for obj_key in ['rvs','pfs','params']:
        if not obj_key in obj_dict: continue
        for obj in obj_dict[obj_key]:
            if hasattr(obj, "isuptodate"): obj.isuptodate = False
            if hasattr(obj, "name") and obj.name != None:
                if obj_key == "rvs" and obj.name in _drvs:
                    update_status(_drvs[obj.name])
                if obj_key == "pfs" and obj.name in _dpfs:
                    update_status(_dpfs[obj.name])
                if obj_key == "params" and obj.name in _dparams:
                    update_status(_dparams[obj.name])
