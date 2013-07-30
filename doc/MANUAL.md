StatsPy Users Guide
===================

.. Contents::

## Introduction

StatsPy is a python based statistical package aimed to help users to build conveniently relatively complex probability functions which can be used later to fit to data, perform statistical tests or extract confidence intervals. StatsPy is originally inspired from the RooFit/RooStats packages used in High Energy Physics within the CERN ROOT analysis framework. The main two motivations for creating a new tool were to build a framework directly from python and thus oriented toward the "pythonic" scripting style, and to create a package built on top of the widely used NumPy/SciPy scientific stack.

The package development is yet at its beginning and currently focused on the main elements to build conveniently a probability function. 

## StatsPy Quick Start

### Conventions

As many packages, StatsPy relies heavily on acronyms. The main ones are used to define the 3 basic classes used in StatsPy.

* **PF = Probability Function**, a generic name, following Kendall's Advanced Theory of Statistics, referring both to Probability Density Functions (PDF/pdf) in the case of continuous random variables and to Probability Mass Functions (PMF/pmf) for discrete random variables.
* **RV = Random Variable**, which is self explanatory.
* **Param**eters denote quantities that appear specifically in the specification of the probability function. In the present package, any function of parameters will be also named a parameter, and such quantities are named *derived* in opposition with *raw* parameters.

### Probability Functions

#### Declaration and evaluation

Probability functions are declared via the base class `statspy.core.PF`

**Example 1**, declare and call a normal distribution:

    >>> import statspy as sp
    >>> mypdf = sp.PF("pdf_norm=norm(x;mu=20,sigma=5)")

In this case the whole PF is defined via the string argument

* `pdf_norm`, the name of the PF is defined on the left-hand side of the assignment statement sign ("="). Giving a name to a PF is optional but recommended. The name can be retrieved via:

        >>> mypdf.name
        pdf_norm

* `norm` is a keyword referring to the `scipy.stats.norm` function which is used to compute the actual value of the PF. Any pdf or pmf defined in scipy.stats module can be used as a keyword. The statistical function should be followed by parenthesis.
* Within the parenthesis, the first part should be used to define the name of the random variable(s), `x`, in the case above.
* The second part of the parenthesis defines the list of shape parameters used to define a normal distribution, i.e. its mean and its rms. Their order should follow the one used by the `scipy.stats.norm` function. The shape parameters are separated from the random variables via the ";" or "|" characters. If a shape parameter is new, it is automatically declared and registered within some internal dictionary. If a shape parameter has already been declared, the PF is linked toward it. It is possible to retrieve a Parameter via the function `get_obj(name)`:

        >>> mu = sp.get_obj("mu")
        >>> mu.value
        20

Alternatively keyword arguments can be used to set various PF members

    >>> poisson_pmf = sp.PF("poisson(n|lbda)",name="pmf_poisson",lbda=10)

To evaluate the probability function in `x`, the `__call__` operator is used. `x` can be either a float or an array:

    >>> mypdf(25)
    0.048394144903828672
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.asarray(range(400))*0.1
    >>> plt.plot(x,mypdf(x),'r-')
    >>> plt.show()

#### Operations on PFs

#### An interlude about parameters

### Perform a fit to data

### Working with Random Variables

## Statistical inference with StatsPy

### Parameters estimation

The free parameters (i.e. parameters for which `Param.const == False`) of a probability function can be fitted to data via two widely used methods.

* The method of least squares which requires as minimial inputs the x- and y- values of a set of data as shown by the following example:

        >>> import statsy as sp
        >>> xdata = 

* The maximum likelihood estimation

### Making hypothesis tests

### Evaluating confidence intervals
