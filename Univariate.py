'''
Univariate-related functions:
[1] qq_plot
[2] chi2_test
[3] ks_test
[4] UnivariateOutliers
[5] MatchDist
[6] Descriptive
[7] BoxPlot
[8] Compare2samp

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 05-03-2022

'''
import numpy as np, pandas as pd, time
import scipy.stats as stats
from warnings import warn
import collections, numbers, inspect
import ipywidgets as widgets
from IPython.display import display
import multiprocessing
from joblib import Parallel, delayed
from functools import partial
from typing import Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import OrdinalEncoder
from scipy.interpolate import interp1d
import matplotlib.transforms as transforms
from sklearn.neighbors import KernelDensity

plt.rcParams.update({'font.family':'sans-serif'})
plt.rcParams.update({'font.sans-serif':'Hiragino Sans GB'})
plt.rc('axes', unicode_minus=False)

__all__ = ["qq_plot", 
           "chi2_test", 
           "ks_test",
           "UnivariateOutliers" ,
           "MatchDist", 
           "Descriptive",
           "BoxPlot", 
           "Compare2samp"]

def __ContDist__(dist):
    
    '''
    ** Private Function **
    Check and return scipy.stats._continuous_distns
    
    Parameters
    ----------
    dist : str or function, default="norm"
        If dist is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If `dist` is a 
        function, it must have an interface similar to <scipy.stats.
        rv_continuous>.
    
    Returns
    -------
    dist : scipy.stats.rv_continuous
    
    params : dict
        Only available when dist is "rv_frozen", otherwise it defaults 
        to None. params contains shape parameters required for specified 
        distribution with two keys i.e. "args" (positional), and "kwds" 
        (keyword).
    
    dist_name : str
        Name of cont. distribution function under <scipy.stats>. 
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
   
    '''
    # Check whether `dist` is callable or not.
    if isinstance(dist, str):
        # Find all continuous distributions in scipy.stats
        cont_dist = stats.rv_continuous
        cont_dist = [d for d in dir(stats) 
                     if isinstance(getattr(stats, d), cont_dist)]

        if dist not in cont_dist:
            warn(f"There is no <{dist}> under scipy.stats. " 
                 f"<norm> is used instead.", Warning)
            dist = getattr(stats, "norm")
        else: dist = getattr(stats, dist)

    modules = ['_continuous_distns', '_distn_infrastructure']
    md = dist.__module__.split(".")[-1]
    
    if md not in modules:
        warn(f"{dist} is neither continuous distribution" 
             f"nor freezing-parameter function from <scipy.stats>." 
             f" <function scipy.stats.norm> is used instead.", Warning)
        return getattr(stats, "norm"), None, dist.__dict__['name']
    
    elif md == '_distn_infrastructure':
        dname = dist.__dict__['dist'].__dict__['name']
        return dist, {'args':dist.args,'kwds':dist.kwds}, dname
    
    else: return dist, None, dist.__dict__['name']

def qq_plot(x, dist="norm", bins=10):
    
    '''
    Q–Q (quantile-quantile) plot is a probability plot, which is a 
    graphical method for comparing two distributions.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If `dist` is a 
        function, it must have an interface similar to <scipy.stats.
        _continuous_distns>.
    
    bins : int, default=10
        It defines the number of quantile bins between 1st and 99th 
        quantiles.
    
    Returns
    -------
    QQ_plot : collections.namedtuple
        A tuple subclasses with named fields as follow:
        
        r : float 
            Pearson’s correlation coefficient.
        
        statistic : float
            Test statistic given (α, df)= (0.05, N-2).
              
        rmse : float
            Root Mean Square Error, where error is defined as 
            difference between x and theoretical dist.
        
        dist_name : str
            Name of cont. distribution function <scipy.stats>. 
   
        params : tuple
            Tuple of output from <scipy.stats.rv_continuous.fit> i.e. 
            MLEs for shape (if applicable), location, and scale 
            parameters from data.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] https://www.researchgate.net/publication/291691147_A_
           modified_Q-Q_plot_for_large_sample_sizes
    
    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, random_state=0)
    
    See whether x follows normal or uniform distribution.
    >>> qq_plot(x, dist="norm")
    QQ_plot(r=0.9998759256965546, 
    ...     statistic=0.6020687774273007, 
    ...     mse=0.00404352706993056, 
    ...     dist_name='norm', 
    ...     params=(1.9492911213351323, 
    ...             1.9963135546858515))
    
    >>> qq_plot(x, dist="uniform")
    QQ_plot(r=0.9717582798499491, 
    ...     statistic=0.6020687774273007, 
    ...     mse=1.536599729650572, 
    ...     dist_name='uniform', 
    ...     params=(-3.5451855128533003, 
    ...             10.93763361798046))
    
    In this case, "norm" returns higher value of `r` along with 
    smaller value of `mse`, thus we could say that a random variable 
    `x`, has a distribution similar to a normal random distribution, 
    N(μ=2,σ=2). However, visualizing a Q-Q plot is highly recommended 
    as indicators can sometimes be inadequate to conclude "goodness of 
    fit" of both distributions.
    
    '''
    keys = ['r', 'statistic', 'rmse', 'dist', 'params']
    qq = collections.namedtuple('QQ_plot', keys)
    
    # Number of bins must not exceed length of x
    n_bins = min(max(bins,2),len(x))+1
    pj = np.linspace(1, 99, n_bins)
    xj = np.percentile(x, q=pj)
    
    dist, params, dist_name = __ContDist__(dist)
    if params is None: 
        params = dist.fit(x)
        qj = dist.ppf(pj/100, *params)
    else: qj = dist.ppf(pj/100)
    r, _ = stats.pearsonr(xj, qj)
  
    # Calculate test statistic.
    alpha, df = 0.05, min(len(xj),500)-2
    t = stats.t.ppf(1-alpha/2,df)
    statistic = np.sqrt(t**2/(t**2+df))
    
    # A modified Q-Q plot
    rmse = np.sqrt(np.nanmean((xj-qj)**2))
    
    return qq(r=r, statistic=statistic, 
              rmse=rmse, dist=dist_name, 
              params=params)

def ks_test(x, dist="norm"):
    
    '''
    The two-sample Kolmogorov-Smirnov test is a general nonparametric 
    method for comparing two distributions by determining the maximum 
    distance from the cumulative distributions, whose function (`s`) 
    can be expressed as: 
    
                          s(x,m) = f(m,x)/n(m)
    
    where f(m,x) is a cumulative frequency of distribution m given x 
    and n(m) is a number of samples of m. The Kolmogorov–Smirnov 
    statistic for two given cumulative distribution function, a and b 
    is:
    
                       D(a,b) = max|s(x,a) - s(x,b)|
                
    where a ∪ b = {x: x ∈ a or x ∈ b}. The null hypothesis or H0 says 
    that both independent samples have the same distribution.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : str or function, default="norm"
        If dist is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If dist is a 
        function, it must have an interface similar to <scipy.stats.
        _continuous_distns>.
    
    Returns
    -------
    KsTest : collections.namedtuple
        A tuple subclasses with named fields as follow:
        
        statistic : float
            Kolmogorov-Smirnov test statistic
            
        p_value : float
            p-value that corresponds to statistic.
            
        dist_name : str
            Name of cont. distribution function <scipy.stats>. 
   
        params : tuple
            Tuple of output from <scipy.stats.rv_continuous.fit> i.e. 
            MLEs for shape (if applicable), location, and scale 
            parameters from data.

    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, random_state=0)
    
    See whether x follows normal distribution or not.
    >>> ks_test(x, dist="norm")
    KsTest(statistic=0.04627618822251736, 
    ...    pvalue=0.9829477885429552, 
    ...    dist_name='norm', 
    ...    params=(0.008306621282718446, 
    ...            1.0587910687362505))
    
    If α is 5% (0.05), we can not reject the null hypothesis (0.983 > 
    0.05).
    
    '''
    keys = ['statistic', 'pvalue', 'dist', 'params']
    KsTest = collections.namedtuple('KsTest', keys)
  
    dist, params, dist_name = __ContDist__(dist)
    if params is None: 
        params = dist.fit(x)
        cdf = dist(*params).cdf
    else: cdf = dist.cdf
    ks =  stats.kstest(x, cdf)
    return KsTest(statistic=ks.statistic,
                  pvalue=ks.pvalue,
                  dist=dist_name,
                  params=params)

def chi2_test(x, dist='norm', bins=10):

    '''
    In the test of Chi-Square (χ2) for homogeneity of proportion, the 
    null hypothesis says that the distribution of sample data fit a 
    distribution from a certain population or not.

    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : str or function, default="norm"
        If dist is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If dist is a 
        function, it must have an interface similar to <scipy.stats.
        _continuous_distns>.
    
    bins : int or sequence of scalars, default=10
        If bins is an int, it defines the number of equal-sample bins. 
        If bins is a sequence, it defines a monotonically increasing 
        array of bin edges, including the rightmost edge.

    Returns
    -------
    Chi2_Test : collections.namedtuple
        A tuple subclasses with named fields as follow:
        
        chisq : float
            The chi-squared test statistic. 
            
        df : int 
            Degrees of freedom.
            
        p_value : float
            p-value that corresponds to chisq.
            
        dist_name : str
            Name of cont. distribution function under <scipy.stats>. 
            
        params : tuple
            Tuple of output from <scipy.stats.rv_continuous.fit> i.e. 
            MLEs for shape (if applicable), location, and scale 
            parameters from data.
          
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
           
    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, random_state=0)
    
    H0 : data follows a normal distribution.
    Ha : data does not follow a normal distribution.
    
    >>> chi2_test(x, dist="norm")
    Chi2_Test(chisq=0.4773224553586323, df=9, 
    ...       pvalue=0.9999750744566653, 
    ...       dist='norm', 
    ...       params=(1.9492911213351323, 1.9963135546858515))
   
    If α is 5% (0.05), we can not reject the null hypothesis (0.99 > 
    0.05). Or we can determine the critical value as follows:
    
    >>> df = 10 - 1
    >>> cv = chi2.ppf(0.95, df)
    16.9190
    
    We cannot reject the null hypotheis since χ2 is 0.4773, which is 
    less than χ2(α=5%, df=10-1) = 16.9190.
    
    '''
    keys = ['chisq', 'df', 'pvalue', 'dist', 'params']
    Chi2Test = collections.namedtuple('Chi2_Test', keys)
    
    if isinstance(bins, int): bins = __quantiles__(x, bins)
    observe = np.histogram(x, bins)[0]/len(x)*100

    # Cumulative density function.
    dist, params, dist_name = __ContDist__(dist)
    if params is None: 
        params = dist.fit(x)
        cdf = dist.cdf(bins, *params)
    else: cdf = dist.cdf(bins)
    expect = np.diff(cdf)*100

    # Chi-squared test statistic and degrees of freedom.
    chisq = ((observe-expect)**2/expect).sum()
    df = max(len(bins[1:])-1,1)
    return Chi2Test(chisq=chisq, df=df,
                    pvalue=1-stats.chi2.cdf(chisq, df=df), 
                    dist=dist_name, params=params)

def __quantiles__(x:np.ndarray, bins:int=10):
    
    '''Create quantile bins'''
    q = np.linspace(0, 100, bins+1)
    bins = np.unique(np.percentile(x, q))
    bins[-1] = bins[-1] + np.finfo(float).eps
    return bins

class UnivariateOutliers():
      
    '''
    This function determines lower and upper bounds on a variable, 
    where any point that lies either below or above those points is 
    identified as outlier. Once identified, such outlier is then 
    capped at a certain value above the upper bound or floored below 
    the lower bound.

    1) Percentile : (α, 100-α)
    2) Sigma : (μ-β.σ, μ+β.σ)
    3) Interquartile Range : (Q1-β.IQR, Q3+β.IQR)
    4) Grubbs' test (Grubbs 1969 and Stefansky 1972) [1] 
    5) Generalized Extreme Studentized Deviate (GESD) [2,3]
    6) Median Absolute Deviation (MAD) [4]
    7) Mean Absolute Error (MAE) [5]
    
    .. versionadded:: 30-05-2021
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm
    .. [2] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
    .. [3] https://towardsdatascience.com/anomaly-detection-with-generalized
           -extreme-studentized-deviate-in-python-f350075900e2
    .. [4] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    .. [5] https://stats.stackexchange.com/questions/339932/iglewicz-and-
           hoaglin-outlier-test-with-modified-z-scores-what-should-i-do-if-t

    Parameters
    ----------
    method : list of str, default=None
        Method of capping outliers i.e. {'iqr', 'mad', 'grubb', 'mae', 
        'sigma', 'gesd', 'pct'}. If None, it defaults to all methods 
        available.
    
    pct_alpha : float, default=0.01
        It refers to the likelihood that the population lies  outside 
        the confidence interval, used in "Percentile".

    beta_sigma : float, default=3.0
        It refers to amount of standard deviations away from its mean, 
        used in "Sigma".

    beta_iqr : float, default=1.5
        Muliplier of IQR (InterQuartile Range).

    grubb_alpha : float, default=0.05
        The significance level, α of Two-sided test, used in "Grubbs' 
        test".
  
    gsed_alpha : float, default=0.05
        The significance level, α of Two-sided test, used in 
        "Generalized Extreme Studentized Deviate", (GSED).

    mad_zscore : float, default=3.5
        Cutoff of modified Z-scores, used in "MAD".
    
    mae_zscore : float, default=3.5
        Cutoff of modified Z-scores, used in "MAE". 
    
    Attributes
    ----------
    limits : collections.namedtuple
        A tuple subclasses with named fields as follow:
        - var    : Variable name
        - method : Univariate-outlier method
        - lower  : Lower bound
        - upper  : Upper bound

    info : pd.DataFrame
        Information table is comprised of:
        - "variable" : variable name
        - "lower"    : Mean of lower bounds
        - "upper"    : Mean of upper bounds
        - "n_lower"  : # of data points below "lower"
        - "n_upper"  : # of data points above "upper"
        - "n_outlier": # of outliers ("n_lower" + "n_upper")
        - "n_notnan" : # of numerical data points
        - "p_outlier": % of total # of outliers

    capped_X : pd.DataFrame
        Capped variables.
    
    exclude : dict
        Excluded variables.
        - non_numeric : List of non-numeric variables.
        - min_numeric : List of variables with number of numerical 
                        values less than defined threshold.
        
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> import pandas as pd
    
    Use the breast cancer wisconsin dataset 
    
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> cols = load_breast_cancer().feature_names
    >>> X = pd.DataFrame(X, columns=cols)
    
    Fit model
    >>> model = UnivariateOutliers().fit(X)
    
    Information table
    >>> model.info
    
    Capped variables
    >>> model.capped_X
    
    '''
    def __init__(self, 
                 pct_alpha:float = 0.01, 
                 beta_sigma:float = 3.0, 
                 beta_iqr:float = 1.5, 
                 grubb_alpha:float = 0.05, 
                 gsed_alpha:float = 0.05, 
                 mad_zscore:float = 3.5, 
                 mae_zscore:float = 3.5, 
                 methods=None):

        # Univariate-outlier functions
        func = {'sigma': __sigma__, 'iqr'  : __iqr__ , 'pct'  : __pct__,
                'grubb': __grubb__, 'gesd' : __gesd__, 'mad'  : __mad__, 
                'mae'  : __mae__ }
        self.func = collections.namedtuple('Univariate_Function', 
                                           func.keys())(*func.values())

        # Univariate-outlier Parameters 
        r_values = [(beta_sigma , 'beta_sigma' , {'>':0, '<':100}),
                    (beta_iqr   , 'beta_iqr'   , {'>':0, '<':100}),
                    (pct_alpha  , 'pct_alpha'  , {'>':0, '<':0.5}),
                    (grubb_alpha, 'grubb_alpha', {'>':0, '<':0.5}),
                    (gsed_alpha , 'gsed_alpha' , {'>':0, '<':0.5}),
                    (mad_zscore , 'mad_zscore' , {'>':0, '<':100}),
                    (mae_zscore , 'mae_zscore' , {'>':0, '<':100})]
        
        kwargs = dict([(a[1], __CheckValue__(*a)) for a in r_values])
        params = collections.namedtuple('Parameters', kwargs.keys())
        self.params = params(**kwargs)
        
        # Check methods.
        default = list(func.keys())
        if methods is None: self.methods = default
        elif np.isin(methods, default).sum()!=len(methods):
            methods = set(methods).difference(default)
            raise ValueError(f'method must be in {set(default)}. '
                             f'Got {methods} instead.')
        else: self.methods = list(methods)
            
    def fit(self, X):
        
        '''
        Fits the model to the dataset `X`.
        
        Parameters
        ----------
        X : array-like or `pd.DataFrame` object
            Sample data.
        
        Attributes
        ----------
        limits : `collections.namedtuple`
            A tuple subclasses with named fields as follow:
            - var    : Variable name
            - method : Univariate-outlier method
            - lower  : Lower bound
            - upper  : Upper bound
        
        info : pd.DataFrame
            Information table is comprised of:
            - "variable" : variable name
            - "lower"    : Mean of lower bounds
            - "upper"    : Mean of upper bounds
            - "n_lower"  : # of data points below "lower"
            - "n_upper"  : # of data points above "upper"
            - "n_outlier": # of outliers ("n_lower" + "n_upper")
            - "n_notnan" : # of numerical data points
            - "p_outlier": % of total # of outliers
        
        capped_X : pd.DataFrame
            Capped variables.
        
        exclude : dict
            Excluded variables.
            - non_numeric : List of non-numeric variables.
            - min_numeric : List of variables with number of numerical 
                            values less than defined threshold.
        
        '''
        # Convert `X` to pd.DataFrame
        X0 = _to_DataFrame(X).copy()
        usecols, not_num, min_num = __Valid__(X0)
        X0 = X0[usecols].values.astype(float)
        self.exclude = {'non_numeric':not_num, 'min_numeric':min_num}
  
        # Initialize paramters
        keys = ['var', 'method', 'lower', 'upper']
        Outlier = collections.namedtuple('Outlier', keys)
        self.limits, i = collections.OrderedDict(), 0
        
        # Loop through all variables and methods.
        for n in range(X0.shape[1]):
            x = X0[~np.isnan(X0[:,n]),n]
            for f,p in zip(self.func._fields, self.params):
                if f in self.methods:
                    limits = getattr(self.func,f)(x, p)
                    args, i = (usecols[n], f) + limits, i + 1
                    self.limits[i] = Outlier(*args)
          
        # Other attributes
        self.__info__(X0)
        self.__cap__(X0, usecols)
        
        return self
    
    def __info__(self, X:pd.DataFrame):
        
        '''
        Univariate-outlier summary.
        
        Attributes
        ----------
        info : pd.DataFrame
            Information table is comprised of:
            - "variable" : variable name
            - "lower"    : Mean of lower bounds
            - "upper"    : Mean of upper bounds
            - "n_lower"  : # of data points below "lower"
            - "n_upper"  : # of data points above "upper"
            - "n_outlier": # of outliers ("n_lower" + "n_upper")
            - "n_notnan" : # of numerical data points
            - "p_outlier": % of total # of outliers
        
        '''
        
        cols=['variable', 'lower', 'upper']
        info = pd.DataFrame([[m.var, m.lower, m.upper] for i,m in 
                             self.limits.items()], columns=cols)
        kwargs = dict(sort=False, as_index=False)
        info = info.groupby(['variable'], **kwargs).mean()
        
        # Number of points that is less than lower bound.
        lower = np.full(X.shape, info['lower'])
        nonan = np.where(np.isnan(X), np.inf, X)
        info['n_lower'] = np.nansum(nonan < lower, axis=0)
        
        # Number of points that is greater than upper bound.
        upper = np.full(X.shape, info['upper'])
        nonan = np.where(np.isnan(X),-np.inf, X)
        info['n_upper'] = np.nansum(nonan > upper,axis=0)
        
        # Other calculated fields.
        info['n_outlier'] = info[['n_lower','n_upper']].sum(axis=1)
        info['n_notnan']  = (~np.isnan(X)).sum(axis=0)
        info['p_outlier'] = info['n_outlier']/info['n_notnan']
        self.info = info
        
    def __cap__(self, X:pd.DataFrame, usecols):
        
        '''Cap X according to calculated limits.'''
        X0 = X.copy()
        lower = np.full(X.shape, self.info['lower'])
        nonan = np.where(np.isnan(X), np.inf, X)
        X0 = np.where(nonan < lower, lower, X0)
        
        upper = np.full(X.shape, self.info['upper'])
        nonan = np.where(np.isnan(X),-np.inf, X)
        X0 = np.where(nonan > upper, upper, X0)
        self.capped_X = pd.DataFrame(X0, columns=usecols)

def __Valid__(X, min_n=10, raise_warning=True):
    
    '''
    ** Private Function **
    Determine variables, whose properties must satisfy following 
    criteria: 
    [1] Data must be numeric or logical, and
    [2] Data must contain numerical values more than `min_n` records.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input array.
    
    min_n : int, default=10
        Minimum number of numerical values in variable.
    
    raise_warning : bool, default=True
        If True, warning is raised.
    
    Returns
    -------
    - usecol  : List of valid variables.
    - not_num : List of non-numeric variables.
    - min_num : List of variables with number of numerical values less 
                than defined threshold.
                
    '''
    # Convert dtype to `float`
    X0 = X.copy()
    for var in X0.columns:
        try: X0[var] = X0[var].astype(float)
        except: pass
    
    not_num = list(set(X0.columns[X0.dtypes.values==object]))
    if len(not_num)>0:
        if raise_warning:
            message = (f'Data variables must be numerical. '
                       f'List of non-numerical variables: {not_num}')
            warn(message)
    usecol = list(set(list(X0)).difference(not_num))
  
    X0 = X0.loc[:,usecol].copy()
    min_num = list(set(X0.columns[(X0.notna())\
                                  .sum(axis=0)<min_n]))
    if len(min_num)>0:
        if raise_warning:
            message = (f'Data variables must contain numerical ' 
                       f'values more than {min_n} records. List ' 
                       f'of invalid variables: {min_num}')
            warn(message)
    usecol = list(set(list(X0)).difference(min_num))
    
    return usecol, not_num, min_num
    
def __Getkwargs__(func):
    
    '''
    ** Private Function **
    Get positional argument(s) from function.
    
    Parameters
    ----------
    func : function
    
    Returns
    -------
    Distionary of parameter names in positional arguments and their 
    default value.
    
    '''
    # Get all parameters from `func`.
    params = inspect.signature(func).parameters.items()
    return dict([(k[1].name, k[1].default) for k in params 
                 if k[1].default!=inspect._empty]) 

def __CheckValue__(x, var='x', r={">":-np.inf,"<": np.inf}) -> float:
    
    '''
    ** Private Function **
    Validate input value (x) whether it satisfies the condition (r). 
    If False, error is raised.
    '''
    fnc = {"==" : [np.equal, "="], 
           "!=" : [np.not_equal, "≠"], 
           ">"  : [np.greater, ">"],
           ">=" : [np.greater_equal, "≥"],
           "<"  : [np.less, "<"], 
           "<=" : [np.less_equal, "≤"]}

    if not isinstance(x, numbers.Number):
        raise ValueError(f'{var} must be numeric. Got {type(x)} instead.')
    elif sum([fnc[k][0](x, r[k]) for k in r.keys()])!=len(r):
        s = ' & '.join([f'{fnc[k][1]} {r[k]}' for k in r.keys()])
        raise ValueError(f'{var} must be {s}. Got {x} instead.')
    else: return x

def _to_DataFrame(X:pd.DataFrame) -> pd.DataFrame:
    
    '''
    ** Private Function **
    If `X` is not `pd.DataFrame`, column(s) will be automatically 
    created with "Unnamed" format.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
    
    '''
    if not (hasattr(X,'shape') or hasattr(X,'__array__')):
        raise TypeError(f'Data must be array-like. Got {type(X)} instead.')
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        try:
            z = int(np.log(X.shape[1])/np.log(10)+1)
            columns = ['Unnamed_{}'.format(str(n).zfill(z)) 
                       for n in range(1,X.shape[1]+1)]
        except: columns = ['Unnamed']
        return pd.DataFrame(X, columns=columns)
    return X

def __iqr__(x, beta=1.5) -> Tuple[float,float]:

    '''
    ** Private Function **
    lower and upper bounds from sample median (interquartile range).
    
    '''
    q1, q3 = np.nanpercentile(x, q=[25,75])
    return q1-(q3-q1)*beta, q3+(q3-q1)*beta
    
def __sigma__(x, beta=3) -> Tuple[float,float]:

    '''
    ** Private Function **
    lower and upper bounds from sample mean (standard deviation).
    
    '''
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return mu-beta*sigma, mu+beta*sigma
  
def __pct__(x, a=0.01) -> Tuple[float,float]:

    '''
    ** Private Function **
    lower and upper bounds from sample median (percentile).
    
    '''
    q = [a*100, 100-a*100]
    return tuple(np.nanpercentile(x, q))

def __grubb__(x, a=0.05) -> Tuple[float,float]:
    
    '''
    ** Private Function **
    lower and upper bounds from sample mean (Grubbs' Test).
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/
           eda35h1.htm
           
    '''
    N = len(x); df = N-2
    ct = stats.t.ppf(a/(2*N), df)
    G = (N-1)/np.sqrt(N)*np.sqrt(ct**2/(ct**2+df))
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return mu-G*sigma, mu+G*sigma

def __gesd__(x, a=0.05) -> Tuple[float,float]:
    
    '''
    ** Private Function **
    Generalized Extreme Studentized Deviate
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/
           eda35h3.htm
    .. [2] https://towardsdatascience.com/anomaly-detection-with-
           generalized-extreme-studentized-deviate-in-python-f350075900e2
           
    '''
    x0, i = x.copy(), 1
    while True:
        r, index = __gesdRstat__(x0)
        cv, val = __lambda__(x0, a), x0[index]
        if (r > cv) & (len(x0)>=15):
            x0, i = np.delete(x0, index), i + 1
        else: break
    return min(x0), max(x0)

def __gesdRstat__(x) -> Tuple[float,int]:
    
    '''
    ** Private Function **
    GESD's r test statistics
    '''
    dev = abs(x-np.mean(x))
    r_stat = max(dev)/max(np.std(x),np.finfo(float).eps)
    return r_stat, np.argmax(dev)

def __lambda__(x, a=0.05) -> float:
    
    '''
    ** Private Function **
    r critical value is computed given the r test statistics.
    
    '''
    N = len(x); df = N-2
    ct = stats.t.ppf(1-a/(2*N),df)
    return (N-1)/np.sqrt(N)*np.sqrt(ct**2/(ct**2+df))

def __mad__(x, cv=3.5) -> Tuple[float,float]:
    
    '''
    ** Private Function **
    Median Absolute Deviation (MAD)
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/
           eda35h.htm    
    '''
    # Calculate modified Z-score.
    z = (x-np.mean(x))/max(np.std(x),np.finfo(float).eps)
    div = abs(z - np.median(z))
    MAD, mz = np.median(div), np.full(len(x),0.)
    if MAD>0: mz = 0.6745*(z-np.median(z))/MAD 
    
    # Select x, whose absolute modified Z-score stays within critical value.
    x0 = np.delete(x, np.arange(len(x))[abs(mz)>cv])
    return min(x0), max(x0)

def __mae__(x, cv=3.5) -> Tuple[float,float]:
    
    '''
    ** Private Function **
    Mean Absolute Error (MAE)
    
    References
    ----------
    .. [1] https://stats.stackexchange.com/questions/339932/iglewicz-
           and-hoaglin-outlier-test-with-modified-z-scores-what-should
           -i-do-if-t     
    '''
    # Calculate modified Z-score.
    z = (x-np.mean(x))/max(np.std(x),np.finfo(float).eps)
    div = abs(z - np.median(z))
    MAE, mz = np.mean(div) ,np.full(len(x),0.)
    if MAE>0: mz = (z-np.median(z))/(1.253314*MAE)
    
    # Select x, whose absolute modified Z-score stays within critical value.
    x0 = np.delete(x, np.arange(len(x))[abs(mz)>cv])
    return min(x0), max(x0)

class MatchDist():
    
    '''
    Finding most-fitted distribution given `X`.
    
    Parameters
    ----------
    dist : list of str or function, default=None
        If item in dist is a string, it defines the name of continuous 
        distribution function under <scipy.stats.rv_continuous>. If 
        item is a function, it must have an interface similar to 
        <scipy.stats>. If None, it defaults to {"norm", "uniform", 
        "expon", "chi2", "dweibull", "lognorm", "gamma", "exponpow", 
        "tukeylambda", "beta"}.
    
    bins : int, default=20
        `bins` defines the number of quantile bins, and is used in 
        "chi2_test", and "ks_test".
    
    n_jobs : int, default=None
        The number of jobs to run in parallel. None means 1. -1 means 
        using all processors.
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/
           eda366.htm
    
    Attributes
    ----------
    result : collections.OrderedDict
        The order of keys is arranged according to input variable. 
        Within each key, there are 3 fields representing method that 
        is used to determine shape of distribution along with its 
        corresponding results, which are:
        - "chi2": Chi-Square test, <function chi_test>
        - "ks"  : Kolmogorov-Smirnov test, <function ks_test>
        - "qq"  : QQ-plot, <function qq_plot>

    info : pd.DataFrame
        Information table is comprised of:
        - "variable"    : variable name
        - "chi2_chisq"  : Chi-Squar test statistic 
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_dist"   : scipy.stats.rv_continuous
        - "ks_statistic": Kolmogorov-Smirnov test statistic 
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "ks_dist"     : scipy.stats.rv_continuous
        - "qq_r"        : QQ-plot correlation
        - "qq_rmse"     : QQ-plot Root Mean Square Error
        - "qq_dist"     : scipy.stats.rv_continuous
    
    hist : dict
        The key is variable name and value is <namedtuple>, "density", 
        whose fields are "hist", "chi2", "ks", and "qq". In each field, 
        there are also sub-fields, which are "x", "y", and "label".
        
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> import pandas as pd
    
    Use the breast cancer wisconsin dataset 
    
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> cols = load_breast_cancer().feature_names
    >>> X = pd.DataFrame(X, columns=cols)
    
    Fit model
    >>> model = MatchDist().fit(X)
    
    Information table
    >>> model.info
    
    Result
    >>> model.result
    
    Histogram data
    >>> model.hist
    '''
    def __init__(self, dist=None, bins=20, n_jobs=None):
        
        if dist is None:
            dist = ["norm", "uniform", "expon", "chi2", 
                    "dweibull", "lognorm", "gamma", "exponpow", 
                    "tukeylambda", "beta"]
        self.dist = self.__ScipyFunction__(dist)
        self.bins = int(__CheckValue__(bins, 'bins', {">":2}))
        
        # Number of processors required
        n_jobs = max(1,n_jobs) if isinstance(n_jobs, int) else 1
        self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
           
    def __ScipyFunction__(self, contdist):
        
        '''scipy.stats._continuous_distns'''
        func = dict()
        for d in contdist:
            key_vals = __ContDist__(d)
            if key_vals not in func.values():
                func[len(func)] = key_vals
        return func
    
    def __rvfrozen__(self, x, key:str):
        
        '''scipy.stats._distn_infrastructure.rv_frozen'''
        dist, params, _ = self.dist[key]
        if params is None: dist = dist(*dist.fit(x)) 
        return dist
    
    def args(self, m:collections.OrderedDict):
        
        '''Change params format'''
        params = m._asdict().get('params')['args']
        return m._replace(params=params)
        
    def fit(self, X):
        
        '''
        Fits the model to the dataset `X`.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Sample data.
 
        Attributes
        ----------
        result : collections.OrderedDict
            The order of keys is arranged according to input variable. 
            Within each key, there are 3 fields representing method that 
            is used to determine shape of distribution along with its 
            corresponding results, which are:
            - "chi2": Chi-Square test, <function chi_test>
            - "ks"  : Kolmogorov-Smirnov test, <function ks_test>
            - "qq"  : QQ-plot, <function qq_plot>

        info : pd.DataFrame
            Information table is comprised of:
            - "variable"    : variable name
            - "chi2_chisq"  : Chi-Squar test statistic 
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_dist"   : scipy.stats.rv_continuous
            - "ks_statistic": Kolmogorov-Smirnov test statistic 
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "ks_dist"     : scipy.stats.rv_continuous
            - "qq_r"        : QQ-plot correlation
            - "qq_rmse"     : QQ-plot Root Mean Square Error
            - "qq_dist"     : scipy.stats.rv_continuous
            
        hist : dict
            The key is variable name and value is <namedtuple>, "density", 
            whose fields are "hist", "chi2", "ks", and "qq". In each field, 
            there are also sub-fields, which are "x", "y", and "label".
        
        '''
        # Convert `X` to pd.DataFrame
        X0 = _to_DataFrame(X).copy()
        usecols, not_num, min_num = __Valid__(X0)
        self.X = X0[usecols].copy()
        X0 = self.X.values.astype(float).copy()
        self.exclude = {'non_numeric':not_num, 'min_numeric':min_num}
           
        # Initialize paramters
        t = widgets.HTMLMath(value='Initializing . . .')
        display(widgets.HBox([t])); time.sleep(1)
        
        # `collections`
        result = collections.OrderedDict()
        method = collections.namedtuple('Methods', ['chi2', 'ks', 'qq'])
        progress = "Calculating . . . {:,.0%}".format
        
        # Set partial functions.
        part_qq = partial(qq_plot  , bins=self.bins)
        part_ch = partial(chi2_test, bins=self.bins)
        rvs_job = Parallel(n_jobs=self.n_jobs)
        mod_job = Parallel(n_jobs=1)
        
        # Loop through all variables and methods.
        for n in range(X0.shape[1]):
            x = X0[~np.isnan(X0[:,n]),n]
            rv_frozen = rvs_job(delayed(self.__rvfrozen__)(x, key) 
                                for key in self.dist.keys())
            
            # QQ-plot
            outs = [delayed(part_qq)(x, dist=rv) for rv in rv_frozen]
            qq = self.args(min(mod_job(outs), key=lambda x : x.rmse))
            
            # Kolmogorov-Smirnov test
            outs = [delayed(ks_test)(x, dist=rv) for rv in rv_frozen]
            ks = self.args(min(mod_job(outs), key=lambda x: x.statistic))
            
            # Chi2 test
            outs = [delayed(part_ch)(x, dist=rv) for rv in rv_frozen]
            chi2 = self.args(min(mod_job(outs), key=lambda x : x.chisq))
            
            result[usecols[n]] = method(chi2, ks, qq)
            t.value = progress((n+1)/X0.shape[1])
            
        # Create attributes
        self.result = result
        self.__info__()
        self.__density__(pd.DataFrame(X0, columns=usecols))
        
        t.value = ""
        return self
                    
    def __info__(self):
            
        '''
        Summary of results.

        Attributes
        ----------
        info : pd.DataFrame
            Information table is comprised of:
            - "variable"    : variable name
            - "chi2_chisq"  : Chi-Squar test statistic 
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_dist"   : scipy.stats.rv_continuous 
            - "ks_statistic": Kolmogorov-Smirnov test statistic 
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "ks_dist"     : scipy.stats.rv_continuous
            - "qq_r"        : QQ-plot correlation
            - "qq_rmse"     : QQ-plot Root Mean Square Error
            - "qq_dist"     : scipy.stats.rv_continuous
                              
        '''
        # Field names
        fields = {'chi2': ['chisq','pvalue','dist'],
                  'ks'  : ['statistic','pvalue','dist'],
                  'qq'  : ['r','rmse','dist']}

        # List of ouputs by variable.
        info = list()
        for var in self.result.keys():
            data = [getattr(self.result[var], m)._asdict()[fld]
                    for m in fields.keys() for fld in fields[m]]
            info.append([var] + data)

        # Columns
        cols = [f'{m}_{fld}' for m in fields.keys() for fld in fields[m]]
        self.info = pd.DataFrame(info, columns=['variable'] + cols)
    
    def __density__(self, X):
    
        '''
        Probability Density Function plot.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Sample data.
        
        Attributes
        ----------
        hist : dict
            The key is variable name and value is <namedtuple>, "density", 
            whose fields are "hist", "chi2", "ks", and "qq". In each field, 
            there are also sub-fields, which are "x", "y", and "label".
            
        '''
        # Initialize parameters.
        self.hist = dict()
        data = collections.namedtuple('data', ["x", "y", "label"])
        density = collections.namedtuple('density', ["hist", "chi2", "ks", "qq"])

        for key in self.result.keys():
        
            # Histogram of `x`
            x = X.loc[X[key].notna(), key].values
            hist, bin_edges = np.histogram(x, self.bins)
            bins = bin_edges[:-1] + np.diff(bin_edges)/2
            attrs = {"hist": data(x=bin_edges, y=hist, label="histogram")}

            for m in ["chi2", "ks", "qq"]:

                # Use <scipy.stats> model along with its paramerters 
                # from self.result .
                r = getattr(self.result[key], m)
                params, dist = r.params, getattr(stats, r.dist)

                # Determine probability density from `x`, and rescale 
                # to match with histogram.
                pdf = dist(*params).pdf(bins)
                pdf = (pdf - min(pdf))/(max(pdf) - min(pdf))
                pdf = pdf*(max(hist) - min(hist)) + min(hist)

                # Label
                d_name = getattr(r, 'dist') 
                stats_ = getattr(r, "pvalue" if m!="qq" else "rmse")
                label = '{:} ({:}, {:,.4f})'.format(m, d_name, stats_)
                attrs[m] = data(x=bins, y=pdf, label=label)

            self.hist[key] = density(*attrs.values())

    def plotting(self, var=None, methods=None, ax=None, 
                 colors=None, tight_layout=True):
    
        '''
        Function to plot PDF.

        Parameters
        ----------
        var : str, default=None
            Variable name in `hist`. If None, the first key is selected.

        methods : list of str, default=None 
            List of statistical methods i.e. "chi2", "ks", and "qq". If
            None, all are selected.

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created with 
            default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than or equal to 3. If 
            None, it uses default colors from Matplotlib.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        Returns
        -------
        ax : Matplotlib axis object

        '''
        args = (self.X, self.hist, var, methods, ax, colors, tight_layout)
        ax = plotting_matchdist_base(*args)
        return ax

def plotting_matchdist_base(X, hist, var=None, methods=None, ax=None, 
                            colors=None, tight_layout=True):
    
    '''
    Function to plot PDF.

    Parameters
    ----------
    X : array-like or pd.DataFrame
        Sample data.
            
    hist : dict
        The key is variable name and value is <namedtuple>, "density", 
        whose fields are "hist", "chi2", "ks", and "qq". In each field, 
        there are also sub-fields, which are "x", "y", and "label".

    var : str, default=None
        Variable name in `hist`. If None, the first key is selected.
        
    methods : list of str, default=None 
        List of statistical methods i.e. "chi2", "ks", and "qq". If
        None, all are selected.
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, `ax` is created with 
        default figsize.

    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 3. If 
        None, it uses default colors from Matplotlib.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().
        
    Returns
    -------
    ax : Matplotlib axis object
    
    '''
    
    # =============================================================
    # Default ax and colors.
    if ax is None: ax = plt.subplots(figsize=(6.5, 4))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(3)]
              if colors is None else colors)
    # -------------------------------------------------------------
    var_list = list(hist.keys())
    if var is None: var = var_list[0]
    if methods is None: methods = ["chi2", "ks", "qq"]
    # ------------------------------------------------------------- 
    kwds = {'chi2' : dict(lw=2, c=colors[0], ls=(0,(5, 1))),
            'ks'   : dict(lw=2, c=colors[1], ls=(0,(1, 1))),
            'qq'   : dict(lw=2, c=colors[2], ls="-")}
    # -------------------------------------------------------------
    zorder = len(methods)
    for fld in hist[var]._fields:
        r, zorder = getattr(hist[var], fld), zorder - 1
        if fld=='hist':
            x = X.loc[X[var].notna(),var].values
            ax.hist(x, bins=r.x, **dict(facecolor="#f1f2f6", 
                                        edgecolor="grey", lw=0.8, 
                                        zorder=-1, label=var))
    # -------------------------------------------------------------
        elif fld in methods: 
            kwargs = {**{'label': r.label}, **kwds[fld]}
            cubic = interp1d(r.x, r.y, kind = "cubic")
            new_x = np.linspace(r.x.min(), r.x.max(), 500)
            ax.plot(new_x, cubic(new_x), **kwargs)
    # =============================================================
    
    # Set other attributes.
    # =============================================================
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(8))
    ax.tick_params(axis='both', labelsize=11)
    # -------------------------------------------------------------
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    ax.text(1.01, 0, "x", fontsize=13, va='center', ha="left", 
            transform=transform)
    # -------------------------------------------------------------
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    # -------------------------------------------------------------
    ax.legend(edgecolor="grey" , borderaxespad=0.5, markerscale=1, 
              columnspacing=0.3, handletextpad=0.5, loc="best", 
              prop=dict(size=12)) 
    if tight_layout: plt.tight_layout()
    # =============================================================
    
    return ax

class Compare2samp:
    
    '''
    Compare two sets of sample by using Chi-Square test for 
    homogeneity [1,2], Kolmogorov-Smirnov [3,4] tests, and Population 
    Stability Index (PSI) [5].
    
    The methods used are the followings:
    
    (1) Using Chi-Square to test Goodness-of-Fit-Test (χ)
        The goodness of fit test is used to test if sample data fits 
        a distribution from a certain population. Its formula is 
        expressed as:
        
                  χ = ∑{(O(i)-E(i))^2/E(i)}, i ∈ 1,2,…,n
                
        where O(i) and E(i) are observed and expected percentages of 
        ith bin, and n is a number of bins
   
    (2) The two-sample Kolmogorov-Smirnov test
        It is a general nonparametric method for comparing two 
        distributions by determining the maximum distance from the 
        cumulative distributions, whose function (`s`) can be 
        expressed as: 
    
                          s(x,m) = f(m,x)/n(m)
    
        where f(m,x) is a cumulative frequency of distribution m given 
        x and n(m) is a number of samples of m. The Kolmogorov–Smirnov 
        statistic for two given cumulative distribution function, a 
        and b is:
    
                       D(a,b) = max|s(x,a) - s(x,b)|
                
        where a ∪ b = {x: x ∈ a or x ∈ b}. The null hypothesis or H0 
        says that both independent samples have the same distribution.
    
    (3) Population Stability Index (PSI)
    
    =================================================================
    |  PSI Value  |      Inference         |        Action          |
    -----------------------------------------------------------------
    |    < 0.10   | no significant change  | no action required     |
    |  0.1 – 0.25 | small change           | investigation required |       
    |    > 0.25   | Major shift            | need to delve deeper   |
    =================================================================
            
            PSI = ∑{(%A(i)-%E(i))*LOG(%A(i)/%E(i))}, i ∈ 1,2,…,n
        
        In addition, %A(i) and %E(i) can be expressed as:
        
                      %A(i) = A(i)/A, %E(i) = E(i)/E
          
        where "A(i)" and "E(i)" are actual and expected amount of 
        ith bin, and "n" is a number of bins.

    Parameters
    ----------
    bins : dict, default=None
        Key must be column name and its value must be a monotonically 
        increasing array of bin edges, including the rightmost edge.
        This is only applicable when data type is numeric.
        
    global_bins : int, defualt=10
        Number of Chi-Square bins to start off with. This is relevant
        to feature, whose `bins` is not provided.

    equal_width : bool, default=True
        If True, it uses equal-width binning, otherwise equal-sample 
        binning is used instead.
        
    max_category : int, default=100
        If number of unique elements from column with "object" dtype, 
        is less than or equal to max_category, its dtype will be 
        converted to "category". max_category must be greater than or 
        equal to 2.
    
    frac : float, default=0.01
        It defines a minimum fraction (%) of expected samples per bin. 
        A minimum number of samples resulted from `frac` is 5.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Goodness_of_fit
    .. [2] https://courses.lumenlearning.com/wmopen-concepts-
           statistics/chapter/test-of-homogeneity/
    .. [3] https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test
    .. [4] https://docs.scipy.org/doc/scipy/reference/generated/scipy.
           stats.ks_2samp.html
    .. [5] http://www.stat.wmich.edu/naranjo/PSI.pdf
           
    Attributes
    ----------
    result : collections.OrderedDict
        The order of keys is arranged according to input variable. 
        Within each key, it contains "Stats" (collections.namedtuple) 
        with following fields:
        
        - "variable"    : Variable name
        - "chi2_chisq"  : Chi-Square test statistic
        - "chi2_df"     : Chi-Square degrees of freedom
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_bins"   : Chi-Square bin edges
        - "ks_stat"     : Kolmogorov-Smirnov test statistic 
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "psi"         : Population Stability Index
        - "dtype"       : Data type

    info : pd.DataFrame
        Information table is comprised of:
        
        - "variable"    : Variable name
        - "chi2_chisq"  : Chi-Square test statistic
        - "chi2_df"     : Chi-Square degrees of freedom
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_bins"   : Number of Chi-Square bins
        - "ks_stat"     : Kolmogorov-Smirnov test statistic
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "psi"         : Population Stability Index
        - "dtype"       : Data type
    
    '''
    def __init__(self, bins=None, global_bins=10, equal_width=True, 
                 max_category=100, frac=0.01):
        
        self.bins = bins if bins is not None else dict()
        self.global_bins = global_bins
        self.equal_width = equal_width
        self.max_category = max_category
        self.frac = min(np.fmax(frac, np.finfo("float32").eps), 0.9)
    
    def fit(self, X1, X2, use_X1=True):
        
        '''
        Fit model.
        
        Parameters
        ----------
        X1, X2 : array-like or pd.DataFrame
            Two DataFrames of sample observations, where their sample 
            sizes can be different but they must have the same number of 
            features (columns).
            
        use_X1 : bool, default=True
            If True, it uses X1, and X2 as expected and observed samples, 
            respectively, and vice versa when use_X1 is False.
    
        Attributes
        ----------
        result : collections.OrderedDict
            The order of keys is arranged according to input variable. 
            Within each key, it contains "Stats" (collections.namedtuple) 
            with following fields:
            
            - "variable"    : Variable name
            - "chi2_chisq"  : Chi-Square test statistic
            - "chi2_df"     : Chi-Square degrees of freedom
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_bins"   : Chi-Square bin edges
            - "ks_stat"     : Kolmogorov-Smirnov test statistic 
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "psi"         : Population Stability Index
            - "dtype"       : Data type

        info : pd.DataFrame
            Information table is comprised of:
            
            - "variable"    : Variable name
            - "chi2_chisq"  : Chi-Square test statistic
            - "chi2_df"     : Chi-Square degrees of freedom
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_bins"   : Number of Chi-Square bins
            - "ks_stat"     : Kolmogorov-Smirnov test statistic
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "psi"         : Population Stability Index
            - "dtype"       : Data type
        
        '''
        
        # =============================================================
        # Convert `X` to pd.DataFrame
        X1 = _to_DataFrame(X1).copy()
        X2 = _to_DataFrame(X2).copy()
        # -------------------------------------------------------------
        # Assign to expected and observed samples
        x_exp, x_obs = (X1, X2) if use_X1 else (X2, X1)
        x_exp = column_dtype(x_exp, self.max_category)
        x_obs = column_dtype(x_obs, self.max_category)
        self.n_ = [x_exp.shape[0], x_obs.shape[0]]
        n_min = max(int(self.frac * x_exp.shape[0]), 5)
        # -------------------------------------------------------------
        # Numeric and catigorical features from expected observation
        cat_features = list(x_exp.select_dtypes(include="category"))
        num_features = list(x_exp.select_dtypes(include=np.number))
        # -------------------------------------------------------------
        # Initialize parameters.
        self.fields = ["chi2_chisq", "chi2_df", "chi2_pvalue", 
                       "chi2_bins", "ks_stat", "ks_pvalue", "psi", 
                       "dtype"]
        Stats = collections.namedtuple('Stats', self.fields)   
        self.result = collections.OrderedDict()
        # =============================================================

        # Numerical Features
        # =============================================================
        fields = ["f_exp", "f_obs", "type"]
        Params = collections.namedtuple('Params', fields)
        self.hist_data = collections.OrderedDict()
        # -------------------------------------------------------------
        for feat in num_features:
            data1 = x_exp[feat].values.copy()
            data2 = x_obs[feat].values.copy()
        # -------------------------------------------------------------
            # Calculate bin edges, given binning method.
            chi2_bins = self.bins.get(feat, None)
            if chi2_bins is None:
                args = (data1, self.global_bins, self.equal_width)
                chi2_bins = self.__bins__(*args)
            chi2_bins = self.__leqx__(data1, chi2_bins, n_min)
        # -------------------------------------------------------------
            # Frequency of expected and observed samples
            f_exp = self.__freq__(data1, chi2_bins)
            f_obs = self.__freq__(data2, chi2_bins)
            #f_exp = np.where(f_exp==0, np.finfo(float).eps, f_exp)
            f_exp = np.where(f_exp==0, 0.5/len(data1), f_exp)
        # -------------------------------------------------------------
            # Chi-Square test for goodness of fit.
            chi2_chisq = ((f_obs-f_exp)**2/f_exp).sum()*100
            chi2_chisq = min(chi2_chisq, 999) #<== limit
            chi2_df = max(len(chi2_bins)-2,1)
            chi2_pvalue = 1-stats.chi2.cdf(chi2_chisq, df=chi2_df)
        # -------------------------------------------------------------
            # Kolmogorov-Smirnov test for goodness of fit.
            kwd = dict(alternative='two-sided', mode='auto')
            ks_stat, ks_pvalue = stats.ks_2samp(data1, data2, **kwd)
        # -------------------------------------------------------------   
            # Population Stability Index
            f_obs = np.where(f_obs==0, np.finfo(float).eps, f_obs)
            psi = (f_obs - f_exp) * np.log(f_obs / f_exp)
            psi = np.nan_to_num(psi, nan=0.).sum()
        # -------------------------------------------------------------   
            self.hist_data[feat] = Params(*(f_exp, f_obs, "number"))
            self.result[feat] = Stats(*(chi2_chisq, chi2_df, chi2_pvalue, 
                                        chi2_bins, ks_stat, ks_pvalue, 
                                        psi, data1.dtype))
        # =============================================================
            
        # Categorical Features
        # =============================================================
        # Keyword argument for OrdinalEncoder
        kwds = dict(categories='auto', dtype=np.int32, unknown_value=-1,
                    handle_unknown="use_encoded_value")
        # -------------------------------------------------------------
        # Fit and transform data.
        encoder = OrdinalEncoder(**kwds).fit(x_exp[cat_features])
        cat_exp = encoder.transform(x_exp[cat_features])
        cat_obs = encoder.transform(x_obs[cat_features])
        # -------------------------------------------------------------
        for n,feat in enumerate(cat_features):
            data1 = cat_exp[:,n].copy()
            data2 = cat_obs[:,n].copy()
        # -------------------------------------------------------------    
            # Calculate bin edges, given binning method.
            args = (data1, max(data1) + 1, self.equal_width)
            chi2_bins = self.__bins__(*args)
            chi2_bins = self.__leqx__(data1, chi2_bins, n_min)
        # -------------------------------------------------------------    
            # Frequency of expected and observed samples
            f_exp = self.__freq__(data1, chi2_bins, len(chi2_bins))
            f_obs = self.__freq__(data2, chi2_bins, len(chi2_bins))
            f_exp = np.where(f_exp==0, np.finfo(float).eps, f_exp)
        # -------------------------------------------------------------   
            # Chi-Square test for goodness of fit.
            chi2_chisq = ((f_obs-f_exp)**2/f_exp).sum()*100
            chi2_df = max(len(chi2_bins)-2,1)
            chi2_pvalue = 1-stats.chi2.cdf(chi2_chisq, df=chi2_df)
        # -------------------------------------------------------------    
            # Change chi2_bins format 
            # i.e. (Group(x), [element(x,1), .., element(x,n)])
            index = np.digitize(np.arange(0, max(data1)+1), chi2_bins)
            chi2_bins = [(i,list(encoder.categories_[n][index==i])) 
                         for i in np.unique(index)]
        # -------------------------------------------------------------   
            # Population Stability Index
            f_obs = np.where(f_obs==0, np.finfo(float).eps, f_obs)
            psi = (f_obs - f_exp) * np.log(f_obs / f_exp)
            psi = np.nan_to_num(psi, nan=0.).sum()
        # -------------------------------------------------------------    
            self.hist_data[feat] = Params(*(f_exp, f_obs, "category"))
            self.result[feat] = Stats(*(chi2_chisq, chi2_df, chi2_pvalue, 
                                        chi2_bins, np.nan, np.nan, psi,
                                        "category"))
        # =============================================================
        
        self.__info__()
        return self
    
    def __info__(self):
        
        '''self.info : pd.DataFrame'''
        data = []
        attr = lambda k,f : getattr(self.result[k],f)
        for key in self.result.keys():
            stats = [attr(key,fld) if fld!='chi2_bins' 
                     else 0 for fld in self.fields]
            data.append([key] + stats)
        info = pd.DataFrame(data, columns=["variable"] + self.fields)
        info['chi2_bins'] = info['chi2_df'] + 1
        self.info = info.set_index("variable")
      
    def __freq__(self, x, bins, max_index=None):
        
        '''
        Determine frequency (include np.nan). This function uses 
        np.digitize() to determine the indices of the bins to which each 
        value in input array belongs. If any values in x are less than 
        the minimum bin edge bins[0], they are indexed for bin 1. Whereas 
        any values that are greater than the maximum bin edge bin[-1] or 
        are np.nan, they are indexed for bin "len(bins)+1" or "max_index".
        
        '''
        if max_index is None: max_index = len(bins) + 1
        indices = np.clip(np.digitize(x, bins), 1, max_index)
        return np.array([sum(indices==n) for n in range(1, max_index)])/len(x)
        
    def __bins__(self, x, bins, equal_width=True):
        
        '''
        According to binning method (equal-width or equal-sample),this 
        function generates 1-dimensional and monotonic array of bins. The 
        last bin edge is the maximum value in x plus np.finfo("float32").
        eps.
        
        '''
        bins = np.fmax(bins, 2) + 1
        if equal_width: 
            args = (np.nanmin(x), np.nanmax(x), bins)
            bins = np.linspace(*args)
        elif equal_width==False:
            q = np.linspace(0, 100, bins)
            bins = np.unique(np.nanpercentile(x, q))
        bins[-1] = bins[-1] + np.finfo("float32").eps
        return bins
        
    def __leqx__(self, x, bins, n_min=5):
    
        '''
        To ensure that the sample size is appropriate for the use of the 
        test statistic, we need to ensure that frequency in each bin must
        be greater than "n_min". Bin is collasped to its immediate left 
        bin if above condition is not met, except the first bin.
        
        '''
        notnan = x[~np.isnan(x)]
        if len(bins)>2:
            while True:
                leq5  = (np.histogram(notnan, bins)[0] < n_min)
                index = np.fmax(np.argmax(leq5),1)
                if sum(leq5)==0: return bins
                else: bins = np.delete(bins, index)
                if len(bins)<3: return bins
        else: return bins

    def plotting(self, var, ax=None, colors=None, labels=None, 
                 tight_layout=True, decimal=0, expect_kwds=None, 
                 observe_kwds=None, xticklabel_format=None, max_display=1):
        
        '''
        Plot Chi-Square Goodness of Fit Test.

        Parameters
        ----------
        var : str
            Variable name in self.info (attribute).

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created with 
            default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than 1. If None, it uses 
            default colors from Matplotlib.
        
        labels : list of str, default=None
            A sequence of strings providing the labels for each class i.e. 
            "0", and "1". If None, it defaults to ["Expect", "Observe"].

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        decimal : int, default=0
            Decimal places for annotation of value(s).

        expect_kwds : keywords, default=None
            Keyword arguments of expected samples to be passed to "ax.bar".

        observe_kwds : keywords, default=None
            Keyword arguments of observed samples to be passed to "ax.bar".

        xticklabel_format : string formatter, default=None
            String formatters (function) for ax.xticklabels values. If None, 
            it defaults to "{:,.xf}".format, where x is a decimal place
            determined automatically by algorithm. If x exceeds 5, it
            defaults to "{:,.1e}".format.

        max_display : int, default=1
            Maximum number of categories to be displayed. This is available 
            only when dtype=="category".

        Returns
        -------
        ax : Matplotlib axis object

        '''
        args = (var, ax, colors, labels, tight_layout, decimal, expect_kwds, 
                observe_kwds, xticklabel_format, max_display)
        ax = plotting_2samp_base(self, *args)
        return ax

def column_dtype(X, max_category=100):
    
    '''
    This function converts columns to best possible dtypes which are 
    "float32", "int32" (boolean), "category", and "object". However, 
    it ignores columns, whose dtype is either np.datetime64 or 
    np.timedelta64.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input array.
    
    max_category : int, default=100
        If number of unique elements from column with "object" dtype, 
        is less than or equal to max_category, its dtype will be 
        converted to "category". max_category must be greater than or 
        equal to 2.
    
    Returns
    -------
    Converted_X : pd.DataFrame
    
    '''
    # Select columns, whose dtype is neither datetimes, nor timedeltas.
    exclude = [np.datetime64, np.timedelta64] 
    columns = list(X.select_dtypes(exclude=exclude))
    
    if isinstance(max_category, int): max_category = max(2, max_category)
    else: max_category = 100
    
    # Replace pd.isnull() with np.nan
    Converted_X = X.copy()
    Converted_X.iloc[:,:] = np.where(X.isnull(), np.nan, X)
    
    for var in columns:
        x = Converted_X[var].copy()
        try:
            float32 = x.astype("float32")
            if np.isnan(float32).sum()==0:
                int32 = x.astype("int32")
                if (int32-float32).sum()==0: Converted_X[var] = int32
                else: Converted_X[var] = float32
            else: Converted_X[var] = float32 
        except:
            objtype = x.astype("object")
            n_unq = len(objtype.unique())
            if n_unq<=max_category:
                Converted_X[var] = x.astype(str).astype("category") 
            else: Converted_X[var] = objtype
    return Converted_X

def plotting_2samp_base(compare, var, ax=None, colors=None, labels=None,
                        tight_layout=True, decimal=0, expect_kwds=None, 
                        observe_kwds=None, xticklabel_format=None, 
                        max_display=2):
        
    '''
    Plot Chi-Square Goodness of Fit Test.

    Parameters
    ----------
    compare : class object
        Fitted `Compare2samp` object.
    
    var : str
        Variable name in self.info (attribute).

    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, `ax` is created with 
        default figsize.

    colors : list of color-hex, default=None
        Number of color-hex must be greater than 1. If None, it uses 
        default colors from Matplotlib.
        
    labels : list of str, default=None
        A sequence of strings providing the labels for each class i.e. 
        "0", and "1". If None, it defaults to ["Expect", "Observe"].
        
    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    decimal : int, default=0
        Decimal places for annotation of value(s).

    expect_kwds : keywords, default=None
        Keyword arguments of expected samples to be passed to "ax.bar".

    observe_kwds : keywords, default=None
        Keyword arguments of observed samples to be passed to "ax.bar".

    xticklabel_format : string formatter, default=None
        String formatters (function) for ax.xticklabels values. If None, 
        it defaults to "{:,.xf}".format, where x is a decimal place
        determined automatically by algorithm. If x exceeds 5, it
        defaults to "{:,.1e}".format.

    max_display : int, default=1
        Maximum number of categories to be displayed. This is available 
        only when dtype=="category".

    Returns
    -------
    ax : Matplotlib axis object

    '''
    # ===============================================================
    # Get values from self.hist_data
    var_list = list(compare.hist_data.keys())
    if var not in var_list:
        raise ValueError(f"var must be in {var_list}. "
                         f"Got {var} instead.")
    f_exp = compare.hist_data[var].f_exp
    f_obs = compare.hist_data[var].f_obs
    dtype = compare.hist_data[var].type
    data  = compare.result[var]
    bins  = data.chi2_bins
    x = np.arange(len(f_exp))
    # ---------------------------------------------------------------
    # Create matplotlib.axes if ax is None.
    width  = max(6.5, len(f_exp) * 0.9)
    if ax is None: ax = plt.subplots(figsize=(width, 4.5))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # ===============================================================

    # Bar plots
    # ===============================================================
    num_format = ("{:,." + str(decimal) + "%}").format
    anno_kwds = dict(xytext =(0,4), textcoords='offset points', 
                     va='bottom', ha='center', fontsize=13, 
                     fontweight='demibold')
    if labels is None: labels = ["Expect", "Observe"]
    # ---------------------------------------------------------------
    # Vertical bar (Expect).
    kwds = dict(width=0.4, alpha=0.9, color=colors[0], 
                label='{} ({:,d})'.format(labels[0], compare.n_[0]))
    ax.bar(x-0.25, f_exp, **({**kwds, **expect_kwds} if 
                             expect_kwds is not None else kwds))
    # ---------------------------------------------------------------
    # Annotation (Expect).
    kwds = {**anno_kwds, **dict(color=colors[0])}
    for xy in zip(x-0.25, f_exp): 
        ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
    # ---------------------------------------------------------------
    # Vertical bar (Observe).   
    kwds = dict(width=0.4, alpha=0.9, color=colors[1], 
                label='{} ({:,d})'.format(labels[1], compare.n_[1]))    
    ax.bar(x+0.25, f_obs, **({**kwds, **observe_kwds} if 
                             observe_kwds is not None else kwds))
    # ---------------------------------------------------------------
    # Annotation (Observe).
    kwds = {**anno_kwds, **dict(color=colors[1])}
    for xy in zip(x+0.25, f_obs): 
        ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
    # ===============================================================

    # Set other attributes         
    # ===============================================================
    title  = f"Variable : {var}\n" 
    pvalue = lambda v : "N/A" if np.isnan(v) else "{:,.0%}".format(v)
    args   = (pvalue(data.chi2_pvalue), pvalue(data.ks_pvalue)) 
    title += r"p-value ($\chi^{2}$, KS) : " + "({}, {})".format(*args)
    title += "\nPSI : {:.4f}".format(data.psi)
    props = dict(boxstyle='square', facecolor='none', alpha=0)
    ax.text(0, 0.97, title, transform=ax.transAxes, fontsize=13,
            va='top', ha="left", bbox=props)
    # ---------------------------------------------------------------
    # x-ticklabels number format
    if (xticklabel_format is None) & (dtype=="number"): 
        pts = 0
        while True:
            adj = np.unique(np.round(bins, pts))
            if len(adj)<len(bins): pts += 1
            else: break
        if pts > 5: n_format = "{:,.1e}".format
        n_format = ("{:,." + str(pts) + "f}").format
    else: n_format = xticklabel_format
    # ---------------------------------------------------------------   
    xticklabels = []
    if dtype=="number":
        for n in np.arange(len(f_exp)):
            if n < len(f_exp)-1: 
                r = r"$<$"+ f"{n_format(bins[n+1])}"
            else: r = (r"$\geq$" + f"{n_format(bins[-1])}" 
                       + "\nor missing")
            xticklabels.append(f"Group {n+1}" + "\n" + r)
    # ---------------------------------------------------------------
    elif dtype=="category":
        for n, m in bins:
            if max_display>0:
                # format = {A, B,...(n)}
                n_set = [f'"{s}"' for s in np.array(m)[:max_display]]
                if len(m)>max_display: 
                    n_set += [" ..({:,d})".format(len(m))]
                xticklabels += [f"Group {n}" + "\n{" + 
                                ",".join(n_set) + "}"]
            else: xticklabels += [f"Group {n}" + 
                                  "\n({:,.0f})".format(len(m))]
    # ---------------------------------------------------------------            
    xticklabels += [label.split('\n')[0] for label in xticklabels]
    xtick_pos = (list(x) + list(x+1e-8))
    plt.xticks(xtick_pos, xticklabels, fontsize=12)
    # ---------------------------------------------------------------
    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.xaxis.get_majorticklabels()
    for i in x: tick_labels[i].set_color("#999999")
    ax.set_xlim(-0.5, len(f_exp)-0.5)
    # ---------------------------------------------------------------
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max/0.7)
    # ---------------------------------------------------------------
    ax.legend(edgecolor="grey", facecolor="none", borderaxespad=0.5, 
              markerscale=1, columnspacing=0.3,handletextpad=0.5, 
              loc="upper right", prop=dict(size=12)) 
    if tight_layout: plt.tight_layout()
    # ===============================================================
    
    return ax

class Descriptive(UnivariateOutliers):
    
    '''
    Generate descriptive statistics, which include those that 
    summarize the central tendency, dispersion and shape of a 
    dataset’s distribution, excluding NaN.
    
    Parameters
    ----------
    methods : list of str, default=None
        Method of capping outliers i.e. {"iqr", "mad", "grubb", "mae", 
        "sigma", "gesd", "pct"}. If None, it defaults to all methods 
        available, except "gesd". See UnivariateOutliers.__doc__ for 
        more details.
        
    plot_kwds : keywords
        Keyword arguments to be passed to kernel density estimate plot 
        <DescStatsPlot>.
    
    Attributes
    ----------
    str_info : pd.DataFrame
        Information table is comprised of:
        - "variable" : Variable name
        - "unique"   : Number of unique items
        - "missing"  : Number of missing values

    num_info : pd.DataFrame
        Summary statistics of Dataframe provided.
            
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> import pandas as pd
    
    Use the breast cancer wisconsin dataset 
    
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> cols = load_breast_cancer().feature_names
    >>> X = pd.DataFrame(X, columns=cols)
    
    Fit model
    >>> model = Descriptive().fit(X)
    
    Summary statistics of numerical X
    >>> model.num_info
    
    Summary statistics of non-numerical X
    >>> model.str_info
    
    Visualize distribution of `var`
    >>> model.plotting(var)
    
    '''
    def __init__(self, methods=None, plot_kwds=None):
        
        if methods is None: 
            self.methods = ["pct", "sigma", "iqr", 
                            "grubb", "mad" , "mae"]
        
        if plot_kwds is not None:
            kwds = __Getkwargs__(DescStatsPlot)
            keys = set(kwds).intersection(plot_kwds.keys())
            if len(keys)>0:
                self.kwds = dict([(k, plot_kwds[k]) for k in keys])
        else: self.kwds = dict()

    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : array-like or `pd.DataFrame` object
            Sample data.
        
        Attributes
        ----------
        str_info : pd.DataFrame
            Information table is comprised of:
            - "variable" : Variable name
            - "unique"   : Number of unique items
            - "missing"  : Number of missing values
            See Descriptive.__str__.__doc__ for more details.
        
        num_info : pd.DataFrame
            Summary statistics of Dataframe provided. See Descriptive.
            __num__.__doc__ for more details.

        References
        ----------
        .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/
               eda35b.htm
               
        '''
        # Convert `X` to pd.DataFrame
        X0 = _to_DataFrame(X).copy()
        numcols, strcols, nancols = __Valid__(X0, 1, False)
        
        # Separate data into numerical and non-numerical.
        X_num = X0[numcols].astype(float).copy()
        X_str = X0[strcols].copy()
        self.exclude = nancols
        del X0
        
        if len(numcols)>0: 
            self.__num__(X_num)
            self.X = X_num.copy()
        else: self.num_info = None
            
        if len(strcols)>0: self.__str__(X_str)
        else: self.str_info = None
        
        return self
    
    def __num__(self, X:pd.DataFrame):
        
        '''
        Summary statistics of numerical X provided.

        Attributes
        ----------
        num_info : pd.DataFrame 
            Information table is comprised of:
            - "variable"   : Variable name
            - "unique"     : Number of unique items
            - "missing"    : Number of missing values
            - "mean"       : Average
            - "std"        : Standard deviation
            - "f_skewness" : Adjusted Fisher-Pearson
            - "g_skewness" : Galton skewness
            - "kurtosis"   : Fisher kurtosis
            - "min"        : Minimum
            - "pct25"      : 25th percentile
            - "pct50"      : 50th percentile (median)
            - "pct75"      : 75th percentile
            - "max"        : Maximum
            - "iqr"        : Interquartile range
            - "lower"      : Outlier lower bound
            - "upper"      : Outlier upper bound
            - "n_lower"    : Number of outliers (< lower)
            - "n_upper"    : Number of outliers (> upper)
        
        '''
        # Order of fields    
        num_fields = ["variable", "unique", "missing", "mean", "std", 
                      "f_skewness", "g_skewness", "kurtosis", 
                      "min", "pct25", "pct50", "pct75", "max", "iqr", 
                      "lower", "upper", "n_lower", "n_upper"]
        
        # Initialize Parameters
        self.num_info_ = collections.OrderedDict()
        data = collections.namedtuple('DescStats', num_fields)

        # Loop through all columns
        for var in X.columns:
            
            # Exclude `np.nan`
            x = X.loc[X[var].notna(), var].values

            # Percentile, and Galton skewness
            Q = np.percentile(x, np.linspace(0,100,5))
            denom = np.where((Q[3]-Q[1])==0,1,Q[3]-Q[1])
            galton = (Q[1]+Q[3]-2*Q[2])/denom
       
            # Univariate Outliers (exclude "gesd").
            Outs = UnivariateOutliers(methods=self.methods).fit(X[[var]])
            Outs = Outs.info.to_dict('records')[0]
            Outs = dict([(key, Outs[key]) for key in num_fields[-4:]])
            
            desc = dict([("variable", var),
                         # Mean and standard deviation
                         ("mean", np.mean(x)), 
                         ("std" , np.std(x)),
                         # Number of unique items and missings
                         ("unique" , pd.Series(x).nunique()), 
                         ("missing", int(np.isnan(X[var]).sum())),
                         # Quartile and IQR
                         ("min"  , Q[0]), ("pct25", Q[1]), 
                         ("pct50", Q[2]), ("pct75", Q[3]), 
                         ("max"  , Q[4]), ("iqr"  , Q[3]-Q[1]),
                         # Adjusted Fisher-Pearson & Galton
                         ("f_skewness", stats.skew(x)), 
                         ("g_skewness", galton),
                         # Fisher kurtosis (normal ==> 0.0)
                         ("kurtosis"  , stats.kurtosis(x))]) 
            self.num_info_[var] = data(**{**desc,**Outs})
            
        # self.num_info
        info, data = self.num_info_, list()
        for key in info.keys():
            data.append(dict([(fld, getattr(info[key],fld)) 
                              for fld in info[key]._fields]))
        
        self.num_info = (pd.DataFrame(data)[num_fields]\
                         .set_index('variable')\
                         .rename(columns={"pct25": "25%",
                                          "pct50": "50%",
                                          "pct75": "75%",
                                          "f_skewness": "fisher skew",
                                          "g_skewness": "galton skew"})\
                         .sort_index().T)
    
    def __str__(self, X:pd.DataFrame):
        
        '''
        Attributes
        ----------
        str_info : pd.DataFrame
            Information table is comprised of:
            - "variable" : Variable name
            - "unique"   : Number of unique items
            - "missing"  : Number of missing values
        
        '''
        str_fields = ["variable", "unique", "missing"]
        info = [dict([("variable" , var),
                      ("unique"   , len(X[var].unique())), 
                      ("missing"  , X[var].isnull().sum())]) 
                for var in X.columns]
        self.str_info = pd.DataFrame(info)[str_fields]\
        .set_index('variable').sort_index().T
        
    def plotting(self, var,  bins="fd", ax=None, colors=None, whis=3.0, 
                 tight_layout=True, hist_kwds=None, plot_kwds=None, 
                 stats_format=None):
        
        '''
        Plot descriptive statistics.
        
        Parameters
        ----------
        var : str, default=None
            Variable name in X.

        bins : int or sequence of scalars or str, default=None
            `bins` defines the method used to calculate the optimal bin 
            width, as defined by <numpy.histogram>. If None, it defaults 
            to "fd".

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created with 
            default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than or equal to 2 i.e. 
            ["Probability Density Function", "Box plot"]. If None, it 
            uses default colors from Matplotlib. 

        whis : float, default=3.0
            It determines the reach of the whiskers to the beyond the 
            first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
            whis*IQR, respectively. This applies to both coordinates and 
            lower and upper bounds accordingly. If None, no bounds are
            determined.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around subplots 
            i.e. plt.tight_layout().

        hist_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.hist".

        plot_kwds : keywords, default=None
            Keyword arguments to be passed to "ax.plot".

        stats_format : string formatter, default=None
            String formatters (function) for statistical values. If None, 
            it defaults to "{:,.3g}".format. This does not include 
            "Skewness".

        Returns
        -------
        ax : Matplotlib axis object

        '''
        kwds = dict(bins=bins, ax=ax, colors=colors, whis=whis, 
                    tight_layout=tight_layout, hist_kwds=hist_kwds, 
                    plot_kwds=plot_kwds, stats_format=stats_format)
        ax = desc_plot_base(self.X, var, **kwds)
        return ax

def desc_plot_base(X, var, bins="fd", ax=None, colors=None, whis=3.0, 
                   tight_layout=True, hist_kwds=None, plot_kwds=None, 
                   stats_format=None):
    
    '''
    Plot descriptive statistics.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input pd.DataFrame object.
        
    var : str
        Variable name in X.
            
    bins : int or sequence of scalars or str, default=None
        `bins` defines the method used to calculate the optimal bin 
        width, as defined by <numpy.histogram>. If None, it defaults 
        to "fd".
        
    ax : Matplotlib axis object, default=None
        Predefined Matplotlib axis. If None, `ax` is created with 
        default figsize.
        
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to 2 i.e. 
        ["Probability Density Function", "Box plot"]. If None, it 
        uses default colors from Matplotlib. 
   
    whis : float, default=3.0
        It determines the reach of the whiskers to the beyond the 
        first and third quartiles, which are Q1 - whis*IQR, and Q3 + 
        whis*IQR, respectively. This applies to both coordinates and 
        lower and upper bounds accordingly. If None, no bounds are
        determined.

    tight_layout : bool, default=True
        If True, it adjusts the padding between and around subplots 
        i.e. plt.tight_layout().

    hist_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.hist".

    plot_kwds : keywords, default=None
        Keyword arguments to be passed to "ax.plot".
        
    stats_format : string formatter, default=None
        String formatters (function) for statistical values. If None, 
        it defaults to "{:,.3g}".format. This does not include 
        "Skewness".
        
    Returns
    -------
    ax : Matplotlib axis object

    '''
    # ===============================================================
    if var not in list(X):
        raise ValueError(f"var must be in {list(X)}. "
                         f"Got {var} instead.")
    x = X.loc[X[var].notna(), var].values.copy()
    # Create matplotlib.axes if ax is None.
    if ax is None: ax = plt.subplots(figsize=(7, 4.8))[1] 
    colors = ([ax._get_lines.get_next_color() for n in range(2)] 
              if colors is None else colors)
    # ---------------------------------------------------------------
    # Calculate `bin_edges`.
    dx = float(np.diff(np.percentile(x, q=[100,0])))*0.01
    amin, amax = min(x)-dx, max(x)+dx
    bin_edges = np.histogram_bin_edges(x, bins, range=(amin,amax)) 
    # ---------------------------------------------------------------
    kwds = dict(facecolor="#f1f2f6", edgecolor="grey", 
                linewidth=0.8, alpha=0.5)
    if hist_kwds is None: hist_kwds = kwds
    else: hist_kwds = {**kwds, **hist_kwds} 
    n, _,_ = ax.hist(x, bin_edges, **{**hist_kwds, **{"zorder":0}})
    ax.axhline(0, lw=1, color="k", zorder=1)
    # ---------------------------------------------------------------
    bandwidth = np.diff(bin_edges)[0]
    z, pdf = __kde__(x, {"bandwidth": bandwidth})
    kwds = dict(color=colors[0], linewidth=2, linestyle="-")
    if plot_kwds is None: plot_kwds = kwds
    else: plot_kwds = {**kwds, **plot_kwds}    
    scale = max(n) / max(pdf)  
    scaled_pdf = pdf*scale
    ax.plot(z, scaled_pdf, **{**plot_kwds, **{"zorder":2}})
    # ---------------------------------------------------------------
    # Minimum and maximum of x
    if stats_format is None: stats_format = "{:,.3g}".format
    args = (ax.transAxes, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    s = [r"N = {:,.5g}".format(len(x)),
         r"min = {}".format(stats_format(np.min(x))),
         r"max = {}".format(stats_format(np.max(x))),
         r"skewness = {:+,.2f}".format(stats.skew(x))]
    ax.text(0, -0.1, ", ".join(s), fontsize=13, 
            va='top', ha="left", transform=transform)
    # ---------------------------------------------------------------
    # Set y-axis for histogram.
    scale = 0.65
    y_min, y_max = ax.get_ylim()
    y_max/= 0.9
    y_min = y_max - (y_max - y_min)/scale
    ax.set_ylim(y_min, y_max)
    # ---------------------------------------------------------------
    # Draw mean of `x` line.
    ax.axvline(np.mean(x), ymin=1-scale, ymax=0.93, color=colors[0], 
               linewidth=1, linestyle="--")
    args = (ax.transData, ax.transAxes)
    transform = transforms.blended_transform_factory(*args)
    s = [r"$\bar{x}$ = " , stats_format(np.mean(x)), 
         r", $\sigma$ = ", stats_format(np.std(x))]
    ax.text(np.mean(x), 1, "".join(s), fontsize=13, va='top', 
            ha="center", transform=transform)
    # ---------------------------------------------------------------
    mean = np.mean(x)
    stdv = np.std(x)
    n_max = np.floor((max(x) - mean)/stdv)
    n_min = np.floor((mean - min(x))/stdv)
    # Coordinates of every standard deviation away from mean.
    x_stdv = np.r_[mean - np.cumsum(np.full(int(n_min), stdv)), 
                   mean + np.cumsum(np.full(int(n_max), stdv))]
    y_stdv = np.interp(x_stdv, xp=z, fp=scaled_pdf)
    for nx,ny in zip(x_stdv, y_stdv): 
        ax.plot([nx]*2, [0,ny], lw=0.8, color=colors[0], 
                solid_capstyle="round", solid_joinstyle="round")
    # ===============================================================

    # Make a box and whisker plot.
    # ===============================================================
    twx_ax = ax.twinx()
    kwds = {0 : dict(widths=0.8 , showfliers=False, 
                     notch=False, vert=False),
            1 : dict(color=colors[1], linewidth=1, linestyle="--"), 
            2 : dict(color=colors[1], linewidth=2)}
    boxplot = twx_ax.boxplot(x, **kwds[0], zorder=0,
                             medianprops =kwds[1], boxprops=kwds[2], 
                             whiskerprops=kwds[2], capprops=kwds[2])
    # ---------------------------------------------------------------
    # Set y-axis for boxplot.   
    y_min, y_max = twx_ax.get_ylim()
    y_min = 0.1
    y_max = (1.8 - y_min) / (1-scale)
    twx_ax.set_ylim(y_min, y_max)
    # ---------------------------------------------------------------
    Q2 = boxplot['medians'][0].get_xdata()[0]
    left  = boxplot['whiskers'][0].get_xdata()
    right = boxplot['whiskers'][1].get_xdata()
    Q1, Q1_iqr = max(left) , min(left)
    Q3, Q3_iqr = min(right), max(right)
    Q1_pct = stats.percentileofscore(x, Q1_iqr, kind="strict")
    Q3_pct = stats.percentileofscore(x, Q3_iqr, kind="strict")
    Q1_pct = "\n(P{:d})".format(int(Q1_pct))
    Q3_pct = "\n(P{:d})".format(int(Q3_pct))
    # ---------------------------------------------------------------
    default = dict(textcoords='offset points', fontsize=13,
                   bbox=dict(facecolor="w", pad=0, edgecolor='none'),
                   arrowprops = dict(arrowstyle = "-"))
    kwds = {"Q2" : dict(xytext=(0,-10), ha="center", va="top"), 
            "Q3" : dict(xytext=(15,15), ha="left", va="bottom"), 
            "Q3*": dict(xytext=(10,0), ha="left", va="center"),
            "Q1" : dict(xytext=(-15,15), ha="right", va="bottom"), 
            "Q1*": dict(xytext=(-10,0), ha="right", va="center")}
    for key in kwds.keys(): kwds[key] = {**kwds[key],**default}
    args = {"Q2" : (stats_format(Q2), (Q2,0.6)), 
            "Q3" : (stats_format(Q3), (Q3,1.0)), 
            "Q3*": (stats_format(Q3_iqr) + Q3_pct, (Q3_iqr,1)),
            "Q1" : (stats_format(Q1), (Q1,1.0)), 
            "Q1*": (stats_format(Q1_iqr) + Q1_pct, (Q1_iqr,1))}  
    # ---------------------------------------------------------------
    twx_ax.annotate(*args['Q2'], **kwds["Q2"])
    if Q1 < Q2: twx_ax.annotate(*args['Q1'], **kwds['Q1'])
    if Q3 > Q2: twx_ax.annotate(*args['Q3'], **kwds['Q3'])
    if Q1_iqr < Q1: twx_ax.annotate(*args['Q1*'], **kwds['Q1*'])
    if Q3_iqr > Q3: twx_ax.annotate(*args['Q3*'], **kwds['Q3*'])
    # ===============================================================

    # Set other attributes         
    # ===============================================================
    if whis is not None:
        Q = np.percentile(x, np.linspace(0,100,5))
        iqr = (Q[3] - Q[1]) * whis
        upper = np.fmin(Q[3] + iqr, max(x))
        lower = np.fmax(Q[1] - iqr, min(x))
        dx = (upper - lower) * 0.01
        ax.set_xlim(lower-dx, upper+dx) 
    # ---------------------------------------------------------------
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
        twx_ax.spines[spine].set_visible(False)
    # ---------------------------------------------------------------   
    ax.yaxis.set_visible(False)
    twx_ax.yaxis.set_visible(False)
    ax.patch.set_alpha(0)
    twx_ax.patch.set_alpha(0)
    # ---------------------------------------------------------------
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    ax.tick_params(axis='x', labelsize=12)
    if tight_layout: plt.tight_layout()
    # ===============================================================

    return ax

def __kde__(x, kernel_kwds=None):

    '''Private function: Kernel Density Estimator'''
    default_kwds = {"bandwidth": 0.2, "kernel": 'gaussian'}
    if kernel_kwds is not None: default_kwds.update(kernel_kwds)
    kde = KernelDensity(**default_kwds).fit(x.reshape(-1,1))

    # score_samples returns the log of the probability density
    z   = np.linspace(min(x), max(x), 101)
    pdf = np.exp(kde.score_samples(z.reshape(-1,1)))
    pdf = (pdf/sum(pdf)).ravel()
    return z, pdf

def BoxPlot(y, x, ax=None, med_fmt=None, colors=None, 
            return_result=False):
    
    '''
    Make a box and whisker plot using Matplotlib boxplot.
    
    Parameters
    ----------
    y : ndarray or pd.Series object
        Input data. Internally, its dtype will be converted 
        to dtype=np.float32.
    
    x : array-like or pd.Series object
        An array of indices, of same shape as y.
    
    ax : Matplotlib axis object, default=None
        Matplotlib axis. If None, ax defaults to figsize of
        (0.8*n_labels, 6), where n_labels is a number of
        unique indices from x.
    
    med_fmt : string formatter, default=None
        String formatters (function) for median values. If 
        None, it defaults to "{:,.2f}".format.
    
    colors : list of color-hex, default=None
        Number of color-hex must be greater than or equal to
        4. The colors are arranged according to parameters as
        follow "boxprops", "whiskerprops", "medianprops", and 
        "flierprops". If None, it uses default colors from 
        Matplotlib.
    
    return_result : bool, default=False
        If True, it returns result from ax.boxplot [1].
    
    References
    ----------
    .. [1] https://matplotlib.org/3.1.1/api/_as_gen/
           matplotlib.axes.Axes.boxplot.html
    
    Returns
    -------
    ax : Matplotlib axis object
    
    result : dict
        A dictionary mapping each component of the boxplot [1].
    
    '''
    # Convert dtype of y to np.float32 
    if isinstance(y, pd.Series): 
        y = y.astype(np.float32).reset_index(drop=True)
    else: y = pd.Series(y.ravel(), name="Value", 
                        dtype=np.float32)
    
    # Convert x to pd.Series
    if not isinstance(x, pd.Series):
        x = pd.Series(x.ravel(), name="Category")
    else: x = x.reset_index(drop=True)
    
    # Define str.format() for median values.
    if med_fmt is None: 
        med_fmt = '{:,.2f}'.format
    elif not callable(med_fmt): 
        med_fmt = '{:,.2f}'.format   

    # Split data according to labels.
    data, labels = [], np.unique(x)
    for label in labels:
        values = y[x==label]
        if len(values)==0: data.append([0])
        else: data.append(values)
    
    if ax is None:
        figsize = (0.8*len(labels), 6)
        ax = plt.subplots(figsize=figsize)[1]
        
    # Default colors
    if colors is None: 
        colors = [ax._get_lines.get_next_color() for n in range(4)] 
    
    # Parameters (ax.boxplot)
    params = dict(notch=False, vert=True, patch_artist=True, labels=labels)
    params.update({"medianprops" : dict(linewidth=1, color=colors[2])})
    params.update({"capprops"    : dict(linewidth=2, color=colors[1])})
    params.update({"whiskerprops": dict(linewidth=2, color=colors[1], 
                                        linestyle="--")})
    params.update({"boxprops"    : dict(linewidth=2)})
    params.update({"flierprops"  : dict(markerfacecolor=colors[3], 
                                        markeredgecolor=colors[3],
                                        markersize=5, marker='s', 
                                        linestyle="none", alpha=0.1)})
    # Box plot
    n_samples = len(y)
    result = ax.boxplot(data, **params)
    xlabelfmt = "{} (N={:,.0f}, Average={:.0%})".format
    ax.set_xlabel(xlabelfmt(x.name,n_samples,1/len(labels)), fontsize=12)
    ax.set_ylabel(y.name, fontsize=12)
    
    # Set patch parameters
    for patch in result["boxes"]:
        patch.set_edgecolor(colors[0])
        patch.set_facecolor("none")
        
    # Parameters (ax.annotate)
    params = dict(textcoords='offset points', xytext=(0,-3), 
                  fontsize=11, va='top', ha='center', color=colors[2])
    params.update({"bbox": dict(boxstyle='square', 
                                pad=0.1, lw=0, fc='white')})
    
    # Annotation (median)
    for x,y in enumerate(data,1): 
        med = np.median(y)
        ax.annotate(med_fmt(med), (x,med), **params)
        ax.scatter(x, med, marker=7, color=colors[2], s=100)
    
    # Label format (ax.set_xticklabels)
    xtickfmt = "{}\n({:.2%})".format
    ax.set_xticklabels([xtickfmt(s,len(a)/n_samples) 
                        for s,a in zip(labels, data)])
    
    if return_result: return ax, result
    else: return ax