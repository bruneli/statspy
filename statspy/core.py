"""module core.py

This module contains the base classes to define a random variable,
a probability density function, or a parameter.
The module is also hosting global dictonaries used to keep track of
the different variables, parameters and pdf declared.

"""

import logging
import operator
import scipy.stats

__all__ = ['RV','PDF','Param','logger']

_drvs    = {}  # Dictionary hosting the list of random variables
_dpdfs   = {}  # Dictionary hosting the list of probability density functions
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
       pdf : statspy.core.PDF
           Probability Density Function object associated to a Random Variable
       params : statspy.core.Param list
           List of shape parameters used to define the pdf
       isuptodate : bool
           Tells whether associated PDF needs to be normalised or not
       logger : logging.Logger
           message logging system

       Examples:
       ---------
       >>> import statspy as spy 
       >>> x = spy.RV("norm(x|mu=10,sigma=2)")
    """

    def __init__(self,*args,**kwargs):
        self.name = ""
        self.pdf = None
        self.params = []
        self.isuptodate = True
        self.logger = logging.getLogger('statspy.core.RV')
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            self._check_kwargs_syntax(kwargs,foundArgs)
        except:
            raise

    def pdf(self,x,**kwargs):
        """Evaluate Probability Density Function in x

        Parameters
        ----------
        x : float
            Random Variable value
        kwargs : dictionary, optional
            Shape parameters values

        """
        # Check if self._pdf contains a pdf() method
        try:
            check_method_exists(obj=self._pdf,name='pdf')
        except:
            raise
        # Get shape parameters
        shape_values = []
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
            shape_values.append(param.value)
        #
        if type(x) == float:
            self.logger.debug('x=%f,shape_values=%s',x,shape_values)
        return self._pdf.pdf(x,*shape_values)

    def _check_args_syntax(self,args):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to PDF without a keyword, it must be a string.")
        # Analyse the string
        theStr = args[0]
        if '=' in theStr:
            self.name = theStr.split('=')[0].strip().lstrip()
            self.logger.debug("Found PDF name %s", self.name)
            theStr = theStr.split('=')[1]
        if not '(' in theStr:
            raise SyntaxError("No pdf found in %s" % theStr)
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
        if not foundArgs and not 'pdf' in kwargs:
            raise SyntaxError("You cannot declare a Random Variable without specifying a pdf.")
        if 'name' in kwargs: self.name = kwargs['name']
        if 'pdf' in kwargs: self.pdf = kwargs['pdf']
        if 'params' in kwargs: self.params = kwargs['params']
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
        return True

    def _declare(self,pdfName,rvName,lpars):
        # Set/Update Random Variable name
        self.name = rvName
        # Declare/Update parameters
        for parStr in lpars:
            parName = parStr.split('=')[0].strip().lstrip()
            parVal = 0.
            if '=' in parStr:
                parVal = float(parStr.split('=')[1].strip().lstrip())
            if not parName in _dparams:
                _dparams[parName] = {'rvs':[],'pdfs':[]}
                _dparams[parName]['obj'] = Param(name=parName,value=parVal)
            if not self.name in _dparams[parName]['rvs']:
                _dparams[parName]['rvs'].append(self.name)
            self.params.append(_dparams[parName]['obj'])
        # Declare pdf (no shape parameter specified yet)
        self._pdf = getattr(scipy.stats,pdfName)
        return

class PDF(object):
    """Base class to define a Probability Density Function. 

       Attributes
       ----------
       name : str (optional)
           Function name
       func : scipy.stats.rv_generic (optional)
           Probability Density Function object associated to a Random Variable
       params : statspy.core.Param list
           List of shape parameters used to define the pdf
       isuptodate : bool
           Tells whether PDF needs to be normalised or not
       logger : logging.Logger
           message logging system

       Examples:
       ---------
       >>> import statspy as spy 
       >>> pdf_n = spy.PDF("poisson(n;mu)",mu=10.)
    """

    def __init__(self,*args,**kwargs):
        self.name = None
        self.func = None
        self.params = []
        self.isuptodate = False
        self.logger = logging.getLogger('statspy.core.PDF')
        self._rvs = []
        try:
            self.logger.debug('args = %s, kwargs = %s',args,kwargs)
            foundArgs = self._check_args_syntax(args)
            self._check_kwargs_syntax(kwargs,foundArgs)
        except:
            raise

    def __call__(self,*args,**kwargs):
        """Evaluate Probability Density Function in x

        Parameters
        ----------
        args : float, ndarray, optional, multiple values for multivariate pdfs
            Random Variable value(s)
        kwargs : dictionary, optional
            Shape parameters values

        """
        # Check if self._pdf contains a pdf() method
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
        if len(rv_values) != len(self._rvs):
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
            self.logger.debug('self.func values=%s', rv_values[0])
        if method_name == 'pmf': 
            return self.func.pmf(rv_values[0], *param_values)
        return self.func.pdf(rv_values[0], *param_values)

    def _check_args_syntax(self,args):
        if not len(args): return False
        if not isinstance(args[0],str):
            raise SyntaxError("If an argument is passed to PDF without a keyword, it must be a string.")
        # Analyse the string
        theStr = args[0]
        if not '(' in theStr:
            raise SyntaxError("No pdf found in %s" % theStr)
        if not ')' in theStr:
            raise SyntaxError("Paranthesis is not closed in %s" % theStr)
        func_name = theStr.split('(')[0].strip()
        if '=' in func_name:
            self.name = func_name.split('=')[0].strip().lstrip()
            self.logger.debug("Found PDF name %s", self.name)
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
        if not foundArgs and not 'func' in kwargs:
            raise SyntaxError("You cannot declare a PDF without specifying a function to caracterize it.")
        if 'func' in kwargs:
            if self.func != None:
                raise SyntaxError("self.func already exists.")
            self.func = kwargs['func']
        for param in self.params:
            if param.name in kwargs and kwargs[param.name] != param.value:
                param.value = kwargs[param.name]
                self.logger.debug('%s value is updated to %f',
                                  param.name,param.value)
        return True

    def _declare(self,func_name,lpars):
        # Declare functional form of PDF (no shape parameter specified yet)
        self.func = getattr(scipy.stats,func_name)
        # Declare/Update parameters
        for parStr in lpars:
            parName = parStr.split('=')[0].strip().lstrip()
            parVal = 0.
            if '=' in parStr:
                parVal = float(parStr.split('=')[1].strip().lstrip())
            if not parName in _dparams:
                _dparams[parName] = {'rvs':[],'pdfs':[]}
                _dparams[parName]['obj'] = Param(name=parName,value=parVal)
            if self.name != None:
                if not self.name in _dparams[parName]['pdfs']:
                    _dparams[parName]['pdfs'].append(self.name)
            else:
                if not self in _dparams[parName]['pdfs']:
                    _dparams[parName]['pdfs'].append(self)
            self.params.append(_dparams[parName]['obj'])
        return

class Param(object):
    """Base class to define a PDF shape parameter. 

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
       bounds : list
           Defines, if necessary, the lower and upper bounds
       formula : list (optional, only for DERIVED parameters)
           List of operators and parameters used to parse an analytic function
       strform : str (optional, only for DERIVED parameters)
           Representation of the formula as a string
       partype : int
           Tells whether it is a RAW or a DERIVED parameter
       isuptodate : bool
           Tells whether value needs to be computed again or not
       logger : logging.Logger
           message logging system

       Examples:
       ---------
       >>> import statspy as spy 
       >>> mu = spy.Param(name="mu",value=10.)
    """

    # Define the different parameter types
    (RAW,DERIVED) = (0,10)

    def __init__(self,*args,**kwargs):
        self.name    = kwargs.get('name',None)
        self.value   = kwargs.get('value',0.)
        self.bounds  = kwargs.get('bounds',[])
        self.formula = kwargs.get('formula',None)
        self.strform = kwargs.get('strform',None)
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
                        strform=Param._build_str_formula(other,'/',self))
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
        _dparams[name] = {'rvs':[],'pdfs':[],'params':[]}
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
    for obj_key in ['rvs','pdfs','params']:
        if not obj_key in obj_dict: continue
        for obj in obj_dict[obj_key]:
            if hasattr(obj, "isuptodate"): obj.isuptodate = False
            if hasattr(obj, "name") and obj.name != None:
                if obj_key == "rvs":
                    update_status(_drvs[obj.name])
                if obj_key == "pdfs":
                    update_status(_dpdfs[obj.name])
                if obj_key == "params":
                    update_status(_dparams[obj.name])