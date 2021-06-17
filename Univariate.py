'''
Univariate-related functions:
[1] qq_plot
[2] chi2_test
[3] ks_test
[4] UnivariateOutliers
[5] MatchDist
[6] Descriptive
[7] DescStatsPlot

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 31-05-2021
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
import matplotlib

__all__ = ["qq_plot", 
           "chi2_test", 
           "ks_test",
           "UnivariateOutliers" ,
           "MatchDist", 
           "Descriptive",
           "DescStatsPlot"]

def qq_plot(x, dist="norm", bins=10):
    
    '''
    Q–Q (quantile-quantile) plot is a probability plot, 
    which is a graphical method for comparing two 
    distributions.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of `float`
        Input data.
    
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of 
        continuous distribution function under <scipy.stats>. 
        If `dist` is a function, it must have an interface 
        similar to <scipy.stats._continuous_distns>.
    
    bins : int, default=10
        `bins` defines the number of quantile bins between
        1st and 99th quantiles.
    
    Returns
    -------
    QQ_plot : collections.namedtuple
        A tuple subclasses with named fields as follow:
        - r : float 
          Pearson’s correlation coefficient.
        - cv : float
          Critival value given α = 0.05 and df = N - 2.
        - rmse : float
          Root Mean Square Error, where error is defined as 
          difference between x and theoretical dist.
        - dist_name : str
          Name of cont. distribution function <scipy.stats>. 
        - params : tuple
          Tuple of output from <scipy.stats.rv_continuous.fit> 
          i.e. MLEs for shape (if applicable), location, and 
          scale parameters from data.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] https://www.researchgate.net/publication/291691147_
           A_modified_Q-Q_plot_for_large_sample_sizes
    
    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    
    >>> kwargs = dict(size=500, random_state=0)
    >>> x = stats.norm(loc=2, scale=2).rvs(**kwargs)
    
    See whether x follows normal or uniform distribution.
    
    >>> qq_plot(x, dist="norm")
    QQ_plot(r=0.9998759256965546, 
    ...     cv=0.6020687774273007, 
    ...     mse=0.00404352706993056, 
    ...     dist_name='norm', 
    ...     params=(1.9492911213351323, 
    ...             1.9963135546858515))
    
    >>> qq_plot(x, dist="uniform")
    QQ_plot(r=0.9717582798499491, 
    ...     cv=0.6020687774273007, 
    ...     mse=1.536599729650572, 
    ...     dist_name='uniform', 
    ...     params=(-3.5451855128533003, 
    ...             10.93763361798046))
    
    In this case, "norm" returns higher value of `r` along with 
    smaller value of `mse`, thus we could say that a random 
    variable `x`, has a distribution similar to a normal random 
    distribution, N(μ=2,σ=2). However, visualizing a Q-Q plot is 
    highly recommended as indicators can sometimes be inadequate 
    to conclude "goodness of fit" of both distributions.
    
    '''
    keys = ['r', 'cv', 'rmse', 'dist', 'params']
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
  
    # Calculate critical value.
    alpha, df = 0.05, min(len(xj),500)-2
    ct = stats.t.ppf(1-alpha/2,df)
    cv = np.sqrt(ct**2/(ct**2+df))
    
    # A modified Q-Q plot
    rmse = np.sqrt(np.nanmean((xj-qj)**2))
    
    return qq(r=r, cv=cv, rmse=rmse,
              dist=dist_name, 
              params=params)

def ks_test(x, dist="norm"):
    
    '''
    The two-sample Kolmogorov-Smirnov test is a general 
    nonparametric method for comparing two distributions by 
    determining the maximum distance from the cumulative 
    distributions, whose function (`s`) can be expressed as: 
    
                      s(x,m) = f(m,x)/n(m)
    
    where `f(m,x)` is a cumulative frequency of distribution 
    `m` given `x` and `n(m)` is a number of samples of `m`.
    The Kolmogorov–Smirnov statistic for two given cumulative 
    distribution function, `a` and `b` is:
    
                   D(a,b) = max|s(x,a) - s(x,b)|
                
    where a ∪ b = {x: x ∈ a or x ∈ b}. The null hypothesis or 
    `H0` says that both independent samples have the same  
    distribution.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : `str` or function, default="norm"
        If `dist` is a string, it defines the name of 
        continuous distribution function under <scipy.stats>. 
        If `dist` is a function, it must have an interface 
        similar to <scipy.stats._continuous_distns>.
    
    Returns
    -------
    KsTest : collections.namedtuple
        A tuple subclasses with named fields as follow:
        - statistic : float
          Critival value
        - p_value : float
          p-value that corresponds to `statistic`.
        - dist_name : str
          Name of cont. distribution function <scipy.stats>. 
        - params : tuple
          Tuple of output from <scipy.stats.rv_continuous.fit> 
          i.e. MLEs for shape (if applicable), location, and 
          scale parameters from data.

    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    
    >>> kwargs = dict(size=500, random_state=0)
    >>> x = stats.norm(loc=2, scale=2).rvs(**kwargs)
    
    See whether x follows normal distribution or not.
    
    >>> ks_test(x, dist="norm")
    KsTest(statistic=0.04627618822251736, 
    ...    pvalue=0.9829477885429552, 
    ...    dist_name='norm', 
    ...    params=(0.008306621282718446, 
    ...            1.0587910687362505))
    
    If α is 5% (0.05), we can not reject the null hypothesis 
    (0.983 > 0.05).
    
    '''
    keys = ['statistic', 'pvalue', 'dist', 'params']
    KsTest = collections.namedtuple('KsTest', keys)
  
    dist, params, dist_name = __ContDist__(dist)
    if params is None: 
        params = dist.fit(x)
        cdf = dist(*param).cdf
    else: cdf = dist.cdf
    ks =  stats.kstest(x, cdf)
    return KsTest(statistic=ks.statistic,
                  pvalue=ks.pvalue,
                  dist=dist_name,
                  params=params)

def chi2_test(x, dist='norm', bins=10):

    '''
    Chi-Square (χ2) is used to test whether sample data 
    fits a distribution from a certain population or not.
    Its null hypothesis or `H0` says that the observed
    population (x) follows the theoretical distribution.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
    \t Input data.
    
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of 
        continuous distribution function under <scipy.stats>. 
        If `dist` is a function, it must have an interface 
        similar to <scipy.stats._continuous_distns>.
    
    bins : int or sequence of scalars, default=10
        If `bins` is an int, it defines the number of 
        equal-sample bins. If `bins` is a sequence, it 
        defines a monotonically increasing array of bin edges, 
        including the rightmost edge.

    Returns
    -------
    Chi2_Test : collections.namedtuple
        A tuple subclasses with named fields as follow:
        - cv : float
          a critival value. If the critical value from the table
          given `degrees of freedom` and `α` (rejection region)
          is less than the computed critical value (`cv`), then
          the observed data does not fit the expected population.
        - df : int 
          Degrees of freedom.
        - p_value : float
          p-value that corresponds to `cv`.
        - dist_name : str
          Name of cont. distribution function under <scipy.stats>. 
        - params : tuple
          Tuple of output from <scipy.stats.rv_continuous.fit> 
          i.e. MLEs for shape (if applicable), location, and scale 
          parameters from data.
    
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
           
    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    
    >>> kwargs = dict(size=500, random_state=0)
    >>> x = stats.norm(loc=2, scale=2).rvs(**kwargs)
    
    H0 : data follows a normal distribution.
    Ha : data does not follow a normal distribution.

    >>> chi2_test(x, dist="norm")
    Chi2_Test(cv=2.3866122767931617, df=7, 
    ...       p_value=0.9353905800003721, 
    ...       dist_name='norm', 
    ...       params=(1.9492911213351323, 1.9963135546858515))
    
    If α is 5% (0.05), we can not reject the null hypothesis 
    (0.94 > 0.05). Or we can determine the critical value as 
    follows:
    
    >>> df = 10 - 2 - 1
    >>> cv = chi2.ppf(0.95, df)
    14.067140449340169
    
    We cannot reject the null hypotheis since χ2 is 2.389, 
    which is less than χ2(α=5%, df=10-2-1) = 14.067
    
    '''
    keys = ['cv', 'df', 'pvalue', 'dist', 'params']
    Chi2Test = collections.namedtuple('Chi2_Test', keys)
    
    if isinstance(bins, int): bins = __quantiles__(x, bins)
    observe = np.histogram(x, bins)[0]

    # Cumulative density function.
    dist, params, dist_name = __ContDist__(dist)
    if params is None: 
        params = dist.fit(x)
        cdf = dist.cdf(bins, *params)
    else: cdf = dist.cdf(bins)
    expect = np.diff(cdf)*len(x)

    # Critical value and degrees of freedom.
    cv = ((observe-expect)**2/expect).sum()
    df = max(len(bins[1:])-2-1,1)
    return Chi2Test(cv=cv, df=df,
                    pvalue=1-stats.chi2.cdf(cv, df=df), 
                    dist=dist_name, params=params)

def __ContDist__(dist):
    
    '''
    Check and return scipy.stats._continuous_distns
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of 
        continuous distribution function under <scipy.stats>. 
        If `dist` is a function, it must have an interface 
        similar to <scipy.stats.rv_continuous>.
    
    Returns
    -------
    dist : scipy.stats.rv_continuous
    
    params : dict
        Only available when `dist` is "rv_frozen", otherwise 
        it defaults to None. `params` contains shape parameters 
        required for specified distribution with two keys i.e. 
        "args" (positional), and "kwds" (keyword).
    
    dist_name : `str`
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

def __quantiles__(x:np.ndarray, bins:int=10):
    
    '''Create quantile bins'''
    q = np.linspace(0, 100, bins+1)
    bins = np.unique(np.percentile(x, q))
    bins[-1] = bins[-1] + np.finfo(float).eps
    return bins

class MatchDist():
    
    '''
    Finding most-fitted distribution given `X`.
    
    Parameters
    ----------
    dist : list of str or function, default=None
        If item in `dist` is a string, it defines the name 
        of continuous distribution function under 
        <scipy.stats.rv_continuous>. If item is a function, 
        it must have an interface similar to <scipy.stats>.
        If None, it defaults to {"norm", "uniform", "expon", 
        "chi2", "dweibull", "lognorm", "gamma", "exponpow", 
        "tukeylambda", "beta"}.
    
    bins : int, default=10
        `bins` defines the number of quantile bins, and is
        used in "chi2_test", and "ks_test".
    
    n_jobs : int, default=None
        The number of jobs to run in parallel. None means 1. 
        -1 means using all processors.
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/
           section3/eda366.htm
    
    Attributes
    ----------
    result : collections.OrderedDict
        The order of keys is arranged according to input
        variable. Within each key, there are 3 fields
        representing method that is used to determine
        shape of distribution along with its corresponding
        results, which are:
        - "chi2": Chi-Square test, <function chi_test>
        - "ks"  : Kolmogorov-Smirnov test, <function ks_test>
        - "qq"  : QQ-plot, <function qq_plot>

    info : pd.DataFrame
        Information table is comprised of:
        - "variable"    : variable name
        - "chi2_cv"     : Chi-Square test critival value
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_dist"   : <scipy.stats.rv_continuous> from 
                          Chi-Square test
        - "ks_statistic": Kolmogorov-Smirnov test critival 
                          value
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "ks_dist"     : <scipy.stats.rv_continuous> from 
                          Kolmogorov-Smirnov test
        - "qq_r"        : QQ-plot correlation
        - "qq_rmse"     : QQ-plot Root Mean Square Error
        - "qq_dist"     : <scipy.stats.rv_continuous> from 
                          QQ-plot

    hist : dict
        The key is variable name and value is <namedtuple>, 
        "density", whose fields are "hist", "chi2", "ks", 
        and "qq". In each field, there are also sub-fields, 
        which are "x", "y", and "label".
        
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
    def __init__(self, dist=None, bins=10, n_jobs=None):
        
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
            The order of keys is arranged according to input
            variable. Within each key, there are 3 fields
            representing method that is used to determine
            shape of distribution along with its corresponding
            results, which are:
            - "chi2": Chi-Square test, <function chi_test>
            - "ks"  : Kolmogorov-Smirnov test, <function ks_test>
            - "qq"  : QQ-plot, <function qq_plot>

        info : pd.DataFrame
            Information table is comprised of:
            - "variable"    : variable name
            - "chi2_cv"     : Chi-Square test critival value
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_dist"   : <scipy.stats.rv_continuous> from 
                              Chi-Square test
            - "ks_statistic": Kolmogorov-Smirnov test critival 
                              value
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "ks_dist"     : <scipy.stats.rv_continuous> from 
                              Kolmogorov-Smirnov test
            - "qq_r"        : QQ-plot correlation
            - "qq_rmse"     : QQ-plot Root Mean Square Error
            - "qq_dist"     : <scipy.stats.rv_continuous> from 
                              QQ-plot

        hist : dict
            The key is variable name and value is <namedtuple>, 
            "density", whose fields are "hist", "chi2", "ks", 
            and "qq". In each field, there are also sub-fields, 
            which are "x", "y", and "label".
        
        '''
        # Convert `X` to pd.DataFrame
        X0 = _to_DataFrame(X).copy()
        usecols, not_num, min_num = __Valid__(X0)
        self.X = X0[usecols].copy()
        X0 = self.X.values.astype(float).copy()
        self.exclude = {'non_numeric':not_num,
                        'min_numeric':min_num}
           
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
            chi2 = self.args(min(mod_job(outs), key=lambda x : x.cv))
            
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
            - "chi2_cv"     : Chi-Square test critival value
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_dist"   : <scipy.stats.rv_continuous> from 
                              Chi-Square test
            - "ks_statistic": Kolmogorov-Smirnov test critival 
                              value
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "ks_dist"     : <scipy.stats.rv_continuous> from 
                              Kolmogorov-Smirnov test
            - "qq_r"        : QQ-plot correlation
            - "qq_rmse"     : QQ-plot Root Mean Square Error
            - "qq_dist"     : <scipy.stats.rv_continuous> from 
                              QQ-plot
        '''
        # Field names
        fields = {'chi2': ['cv','pvalue','dist'],
                  'ks'  : ['statistic','pvalue','dist'],
                  'qq'  : ['r','rmse','dist']}

        # List of ouputs by variable.
        info = list()
        for var in self.result.keys():
            data = [getattr(self.result[var], m)._asdict()[fld]
                    for m in fields.keys() 
                    for fld in fields[m]]
            info.append([var] + data)

        # Columns
        cols = ['variable'] + [f'{m}_{fld}' 
                               for m in fields.keys() 
                               for fld in fields[m]]
        self.info = pd.DataFrame(info, columns=cols)
    
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
            The key is variable name and value is <namedtuple>, 
            "density", whose fields are "hist", "chi2", "ks", 
            and "qq". In each field, there are also sub-fields, 
            which are "x", "y", and "label".
            
        '''
        # Initialize parameters.
        self.hist = dict()
        data = collections.namedtuple('data', ["x","y","label"])
        density = collections.namedtuple('density',
                                         ["hist","chi2","ks","qq"])

        for key in self.result.keys():
        
            # Histogram of `x`
            x = X.loc[X[key].notna(), key].values
            hist, bin_edges = np.histogram(x, "fd")
            bins = bin_edges[:-1] + np.diff(bin_edges)/2
            attrs = {"hist": data(x=bin_edges, y=hist, 
                                  label="histogram")}

            for m in ["chi2","ks","qq"]:

                # Use <scipy.stats> model along with its 
                # paramerters from self.result .
                r = getattr(self.result[key], m)
                params, dist = r.params, getattr(stats, r.dist)

                # Determine probability density from `x`,
                # and rescale to match with histogram.
                pdf = dist(*params).pdf(bins)
                pdf = (pdf-min(pdf))/(max(pdf)-min(pdf))
                pdf = pdf*(max(hist)-min(hist))+min(hist)

                # Label
                d_name = getattr(r, 'dist') 
                stats_ = getattr(r, "pvalue" if m!="qq" else "rmse")
                label = '{:} ({:}, {:,.4f})'.format(m, d_name, stats_)
                attrs[m] = data(x=bins, y=pdf, label=label)

            self.hist[key] = density(*attrs.values())

            
    def plotting(self, var:str, ax=None):
    
        '''
        Function to plot PDF.
        
        Parameters
        ----------
        var : str
            Variable name in X.

        ax : Matplotlib axis, default=None
            If None, `matplotlib.pyplot.axes` is created with 
            default figsize=(6,3.8).
        
        '''
        kwds = {'chi2' : dict(lw=1.5, marker='o', c='#ee5253', fillstyle='none'),
                'ks'   : dict(lw=1.5, marker='s', c='#2e86de', fillstyle='none'),
                'qq'   : dict(lw=1.5, marker='D', c='#10ac84', fillstyle='none'),
                'hist' : dict(color='#c8d6e5', alpha=0.5)}
        
        if ax is None: 
            ax = plt.subplots(figsize=(6,3.8))[1]

        for fld in self.hist[var]._fields:
            r = getattr(self.hist[var], fld)
            kwargs = {**{'label':r.label},**kwds[fld]}
            if fld=='hist':
                x = self.X.loc[self.X[var].notna(),var].values
                ax.hist(x, bins=r.x, **kwargs)
            else: ax.plot(r.x, r.y, **kwargs)

        ax.legend(loc='best', framealpha=0)
        ax.set_title(var, fontsize=14, fontweight ="bold")
        ax.set_ylabel('Numer of Counts', fontweight ="bold")
           
class UnivariateOutliers():
      
    '''
    This function determines lower and upper bounds on 
    a variable, where any point that lies either below or 
    above those points is identified as outlier. Once 
    identified, such outlier is then capped at a certain 
    value above the upper bound or floored below the lower 
    bound.

    1) Percentile : (α, 100-α)
    2) Sigma : (μ-β.σ , μ+β.σ)
    3) Interquartile Range : (Q1-β.IQR, Q3+β.IQR)
    4) Grubbs' test (Grubbs 1969 and Stefansky 1972) 
    5) Generalized Extreme Studentized Deviate (GESD)
    6) Median Absolute Deviation (MAD)
    7) Mean Absolute Error (MAE)
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    method : list of str, default=None
        Method of capping outliers i.e. {'iqr', 'mad', 'grubb', 
        'mae', 'sigma', 'gesd', 'pct'}. If None, it defaults
        to all methods available.
    
    pct_alpha : float, default=0.01
        It refers to the likelihood that the population lies  
        outside the confidence interval, used in "Percentile".

    beta_sigma : float, default=3.0
        It refers to amount of standard deviations away from 
        its mean, used in "Sigma".

    beta_iqr : float, default=1.5
        Muliplier of IQR (InterQuartile Range).

    grubb_alpha : float, default=0.05
        The significance level, α of Two-sided test, used in
        "Grubbs' test".
  
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
        - min_numeric : List of variables with number of 
                        numerical values less than defined 
                        threshold.
        
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
        # elements are arranged in the same order as `func_`
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
        \t Sample data.
        
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
            - min_numeric : List of variables with number of 
                            numerical values less than defined 
                            threshold.
        
        '''
        # Convert `X` to pd.DataFrame
        X0 = _to_DataFrame(X).copy()
        usecols, not_num, min_num = __Valid__(X0)
        X0 = X0[usecols].values.astype(float)
        self.exclude = {'non_numeric':not_num,
                        'min_numeric':min_num}
  
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
    Determine variables, whose properties must satisfy
    following criteria: 
    [1] Data must be numeric or logical, and
    [2] Data must contain numerical values more than 
        `min_n` records.
    
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
    - min_num : List of variables with number of numerical 
                values less than defined threshold.
                
    '''
    # Convert dtype to `float`
    X0 = X.copy()
    for var in X0.columns:
        try: X0[var] = X0[var].astype(float)
        except: pass
    
    not_num = list(set(X0.columns[X0.dtypes.values==object]))
    if len(not_num)>0:
        if raise_warning:
            warn(f'Data variables must be numerical. ' 
                 f'List of non-numerical variables: {not_num}')
    usecol = list(set(list(X0)).difference(not_num))
  
    X0 = X0.loc[:,usecol].copy()
    min_num = list(set(X0.columns[(X0.notna())\
                                  .sum(axis=0)<min_n]))
    if len(min_num)>0:
        if raise_warning:
            warn(f'Data variables must contain numerical ' 
                 f'values more than {min_n} records. ' 
                 f'List of invalid variables: {min_num}')
    usecol = list(set(list(X0)).difference(min_num))
    return usecol, not_num, min_num
    
def __Getkwargs__(func):
    
    '''
    Get positional argument(s) from function.
    
    Parameters
    ----------
    func : function
    
    Returns
    -------
    Distionary of parameter names in positional 
    arguments and their default value.
    
    '''
    # Get all parameters from `func`.
    params = inspect.signature(func).parameters.items()
    return dict([(k[1].name, k[1].default) for k in params 
                 if k[1].default!=inspect._empty]) 

def __CheckValue__(x, var='x', r={">":-np.inf,"<": np.inf}) -> float:
    
    '''
    Validate input value (x) whether it satisfies 
    the condition (r). If False, error is raised.
    '''
    fnc = {"==" : [np.equal, "="], 
           "!=" : [np.not_equal, "≠"], 
           ">"  : [np.greater, ">"],
           ">=" : [np.greater_equal, "≥"],
           "<"  : [np.less, "<"], 
           "<=" : [np.less_equal, "≤"]}

    if not isinstance(x, numbers.Number):
        raise ValueError(f'{var} must be numeric. '
                         f'Got {type(x)} instead.')
    elif sum([fnc[k][0](x, r[k]) for k in r.keys()])!=len(r):
        s = ' & '.join([f'{fnc[k][1]} {r[k]}' for k in r.keys()])
        raise ValueError(f'{var} must be {s}. Got {x} instead.')
    else: return x

def _to_DataFrame(X) -> pd.DataFrame:
    
    '''
    If `X` is not `pd.DataFrame`, column(s) will be
    automatically created with "Unnamed" format.
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    
    '''
    if not (hasattr(X,'shape') or hasattr(X,'__array__')):
        raise TypeError(f'Data must be array-like. ' 
                        f'Got {type(X)} instead.')
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

def __iqr__(x:np.ndarray, beta=1.5) -> Tuple[float,float]:

    '''
    lower and upper bounds from sample median 
    (interquartile range).
    '''
    q1, q3 = np.nanpercentile(x, q=[25,75])
    return q1-(q3-q1)*beta, q3+(q3-q1)*beta
    
def __sigma__(x:np.ndarray, beta:float=3) -> Tuple[float,float]:

    '''
    lower and upper bounds from sample mean 
    (standard deviation).
    '''
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return mu-beta*sigma, mu+beta*sigma
  
def __pct__(x:np.ndarray, a:float=0.01) -> Tuple[float,float]:

    '''
    lower and upper bounds from sample median
    (percentile).
    '''
    q = [a*100, 100-a*100]
    return tuple(np.nanpercentile(x, q))

def __grubb__(x:np.ndarray, a:float=0.05) -> Tuple[float,float]:
    
    '''
    lower and upper bounds from sample mean
    (Grubbs' Test).
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/
           section3/eda35h1.htm
    '''
    N = len(x); df = N-2
    ct = stats.t.ppf(a/(2*N), df)
    G = (N-1)/np.sqrt(N)*np.sqrt(ct**2/(ct**2+df))
    mu, sigma = np.nanmean(x), np.nanstd(x)
    return mu-G*sigma, mu+G*sigma

def __gesd__(x:np.ndarray, a:float=0.05) -> Tuple[float,float]:
    
    '''
    Generalized Extreme Studentized Deviate
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/
           section3/eda35h3.htm
    .. [2] https://towardsdatascience.com/anomaly-
           detection-with-generalized-extreme-studentized-
           deviate-in-python-f350075900e2
    '''
    x0, i = x.copy(), 1
    while True:
        r, index = __gesdRstat__(x0)
        cv, val = __lambda__(x0, a), x0[index]
        if (r > cv) & (len(x0)>=15):
            x0, i = np.delete(x0, index), i + 1
        else: break
    return min(x0), max(x0)

def __gesdRstat__(x:np.ndarray) -> Tuple[float,int]:
    
    '''
    GESD's r test statistics
    '''
    dev = abs(x-np.mean(x))
    r_stat = max(dev)/max(np.std(x),np.finfo(float).eps)
    return r_stat, np.argmax(dev)

def __lambda__(x:np.ndarray, a:float=0.05) -> float:
    
    '''
    Corresponding to the r test statistics, 
    r critical value is computed as follows:
    '''
    N = len(x); df = N-2
    ct = stats.t.ppf(1-a/(2*N),df)
    return (N-1)/np.sqrt(N)*np.sqrt(ct**2/(ct**2+df))

def __mad__(x:np.ndarray, cv:float=3.5) -> Tuple[float,float]:
    
    '''
    Median Absolute Deviation (MAD)
    
    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/
           section3/eda35h.htm
    '''
    # Calculate modified Z-score.
    z = (x-np.mean(x))/max(np.std(x),np.finfo(float).eps)
    div = abs(z - np.median(z))
    MAD, mz = np.median(div), np.full(len(x),0.)
    if MAD>0: mz = 0.6745*(z-np.median(z))/MAD 
    
    # Select x, whose absolute modified Z-score 
    # stays within critical value.
    x0 = np.delete(x, np.arange(len(x))[abs(mz)>cv])
    return min(x0), max(x0)

def __mae__(x:np.ndarray, cv:float=3.5) -> Tuple[float,float]:
    
    '''
    Mean Absolute Error (MAE)
    
    References
    ----------
    .. [1] https://stats.stackexchange.com/questions/339932/
           iglewicz-and-hoaglin-outlier-test-with-modified-
           z-scores-what-should-i-do-if-t     
    '''
    # Calculate modified Z-score.
    z = (x-np.mean(x))/max(np.std(x),np.finfo(float).eps)
    div = abs(z - np.median(z))
    MAE, mz = np.mean(div) ,np.full(len(x),0.)
    if MAE>0: mz = (z-np.median(z))/(1.253314*MAE)
    
    # Select x, whose absolute modified Z-score 
    # stays within critical value.
    x0 = np.delete(x, np.arange(len(x))[abs(mz)>cv])
    return min(x0), max(x0)

class DescStatsPlot():
    
    '''
    Plot descriptive statistics from <Descriptive>.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input array.
    
    num_info : collections.OrderedDict
        An attribute from <Descriptive.fit>.
    
    bw_method : str, scalar or callable, default=0.1
        The method used to calculate the estimator bandwidth. 
        If a scalar, this will be used directly as kde.factor. 
        If a callable, it should take a gaussian_kde instance 
        as only parameter and return a scalar. If None, "scott"
        is used. See References for more details.
    
    use_hist : bool, default=False
        If True, <ax.hist> is used, otherwise <ax.fill_between>.
        Not available when y is provided.
        
    bins : int or sequence of scalars or str, default=None
        `bins` defines the method used to calculate the optimal 
        bin width, as defined by <numpy.histogram>. If None, it
        defaults to "fd".
        
    show_vline : bool, default=True
        If True, it shows 4 vertical lines across the Axes i.e.
        1st quartile, 2nd quartile, 3rd quartile, and mean. 
        Not available when y is provided.

    show_bound : bool, default=True
        If True, it shows lower and upper bounds of outliers.
        Not available when y is provided.
        
    show_stats : bool, default=True
        If True, it shows decriptive statistics on the side of 
        the plot. 
            
    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.stats.gaussian_kde.html
           
    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> import pandas as pd
    
    Use the breast cancer wisconsin dataset 
    
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> cols = load_breast_cancer().feature_names
    >>> X = pd.DataFrame(X, columns=cols)
    
    Get "num_info" from "Descriptive"
    >>> num_info = Descriptive().fit(X).num_info_
    
    Plot "worst smoothness"
    >>> model = DescStatsPlot(X, num_info)
    >>> model.plotting("worst smoothness")
    >>> plt.tight_layout()
    >>> plt.show()
    
    '''
    def __init__(self, X, num_info, 
                 bins = None, 
                 bw_method = 0.1, 
                 use_hist = False, 
                 show_vline = True, 
                 show_stats = True, 
                 show_bound = True):

        self.X = X.copy()
        self.num_info = num_info
        self.bins = "fd" if bins is None else bins
        self.bw_method = bw_method
        self.use_hist = use_hist
        self.show_vline = show_vline 
        self.show_stats = show_stats
        self.show_bound = show_bound
    
    def plotting(self, var, ax=None, xlim=None, y=None):
        
        '''
        Plot histogram
        
        Parameters
        ----------
        var : str
            Variable name in X.

        ax : Matplotlib axis, default=None
            If None, `matplotlib.pyplot.axes` is created with 
            default figsize=(8,3.8).
            
        xlim : tuple(float,float), default=None
            It is the x-axis view limits (left,right), If None, 
            it leaves the limit unchanged.
            
        y : array-like of shape X.shape[0], default=None
        \t An array of labels (int).
            
        '''
        if ax is None: 
            ax = plt.subplots(figsize=(8,3.8))[1] 
            
        dStats = self.num_info[var]
        x = self.X.loc[self.X[var].notna(), var].values.copy()
        if y is None:
            self.__histpdf__(ax, x, dStats, self.bins, 
                             self.bw_method, self.use_hist)
            if self.show_bound: self.__axvline__(ax, dStats)
        else:
            nonan = np.array(y)[self.X[var].notna()].copy()
            self.__yhistpdf__(ax, x, nonan, var, self.bw_method)
        
        if self.show_stats: self.__decstat__(ax, dStats)
        ax.set_title(var, fontsize=14, fontweight ="bold")
        if isinstance(xlim, tuple): ax.set_xlim(*xlim)
            
    def __histpdf__(self, ax, x, dStats, bins="fd",
                    bw_method=0.1, use_hist=False):

        '''Probability Desnsity Function'''
        kwargs = {"hist"  : dict(color="#d1d8e0", bins=bins, alpha=0.7), 
                  "kde"   : dict(color="#4b6584", lw=3),
                  "mean"  : dict(color="#eb3b5a", lw=2),
                  "quant" : dict(color="#3867d6", lw=2), 
                  "left"  : dict(xytext=(-4,0), textcoords="offset points",  
                                 va='center', ha='right', fontsize=14),
                  "right" : dict(xytext=(4,0), textcoords="offset points",
                                 va='center', ha='left', fontsize=14)}

        # Histogram and pdf (kde).
        hist   = np.histogram(x, bins=bins)[0]
        kernel = stats.gaussian_kde(x, bw_method=bw_method)
        unq_x  = np.unique(x)
        pdf    = kernel.pdf(unq_x)

        # Plot histogram and pdf.
        y = self.__rescale__(pdf, hist)
        ax.plot(unq_x, y, **kwargs['kde'])
        if use_hist: ax.hist(x, **kwargs['hist'])
        else: ax.fill_between(unq_x, y, color="#d1d8e0")
        
        if self.show_vline:
            criteria = [("pct25", r"$Q_{1}$", "left" , "quant"),
                        ("pct50", r"$Q_{2}$", "right", "quant"),
                        ("pct75", r"$Q_{3}$", "right", "quant"),
                        ("mean" , r"$\mu$"  , "left" , "mean" )]

            for (fld,s,k0,k1) in criteria:
                x = getattr(dStats, fld)
                y = self.__rescale__(pdf, hist, kernel.pdf(x))[0]
                ax.plot((x,)*2, [0,y], **kwargs[k1])
                ax.annotate(s, (x, y/2), **{**kwargs[k0],
                                            **{'color':kwargs[k1]["color"]}})

        ax.set_ylim(0, ax.get_ylim()[1]*1.1)
        ax.set_ylabel('Numer of Counts', fontweight ="bold") 
    
    def __yhistpdf__(self, ax, x, y, var, bw_method=0.1):

        '''Probability Desnsity Function by class'''
        strfmt = "{} = {:,.0f} ({:,.0%})".format
        labels, cnts = np.unique(y, return_counts=True)
        bins = np.histogram(x, bins="fd")[1]
        
        for c,n in zip(labels,cnts):
            dataset= x[y==c]
            kernel = stats.gaussian_kde(dataset, bw_method)
            hist   = np.histogram(dataset, bins)[0]/len(dataset)
            unq_x  = np.unique(dataset)
            pdf    = kernel.pdf(unq_x)
            
            # Plot histogram and pdf.
            spdf = self.__rescale__(pdf, [0,max(hist)])
            ax.plot(unq_x, pdf, lw=2)
            ax.fill_between(unq_x, pdf, alpha=0.2, 
                            label= strfmt(c,n,n/len(y)))
  
        ax.legend(loc='best', framealpha=0, fontsize=10)
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        for spine in ["right","left","top"]:
            ax.spines[spine].set_visible(False)

    def __axvline__(self, ax, dStats):

        '''Plot ax.axvline'''
        # Initial parameters
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Descriptive statistics
        mu, std = dStats.mean, dStats.std
        lower, upper = dStats.lower, dStats.upper
        p25, p75, iqr =  dStats.pct25, dStats.pct75, dStats.iqr
        out_lines = [(r" Upper " , upper, "#4b6584", (0,-5)), 
                     (r" Lower " , lower, "#4b6584", (0,-5)),
                     (r" $\mu+3\sigma$ ", mu + 3*std, "#eb3b5a", (0,-45)),
                     (r" $\mu-3\sigma$ ", mu - 3*std ,"#eb3b5a", (0,-45)),
                     (r" $Q_{3}+1.5(IQR)$ ", p75 + 1.5*iqr, "#3867d6", (0,-85)),
                     (r" $Q_{1}-1.5(IQR)$ ", p25 - 1.5*iqr, "#3867d6", (0,-85))]  

        # Keyword arguments
        out_kwargs = dict(textcoords="offset points", xytext=(0,0), 
                          va='top', ha='center', fontsize=10, rotation=-90,
                          bbox=dict(facecolor='white', pad=0, ec='none'))

        for (s,v,c,xy) in out_lines:
            if (x_min < v < x_max):
                ax.axvline(v, lw=1, ls='--', color=c)
                ax.annotate(s, (v,y_max), **{**out_kwargs, 
                                             **dict(color=c, xytext=xy)})

    def __decstat__(self, ax, dStats):

        '''Descriptive Statistics'''
        GetVals = lambda t:__numfmt__(getattr(dStats,t))
        text0 = [("Unique", "unique"), ("Missing", "missing"),
                 ("Mean", "mean"), ("Stdev", "std"), 
                 ("Skewness", "f_skewness"), ("Kurtosis", "kurtosis"), 
                 ("Min", "min"), ("25%", "pct25"), ("50%", "pct50"), 
                 ("75%", "pct75"), ("Max", "max")]
        
        s  = ['Count = ' + __numfmt__(self.X.shape[0])]
        s += ['{} = {}'.format(t0, GetVals(t1)) for t0,t1 in text0]
        s += ['{} = {} ({})'.format(t0, GetVals(t1), GetVals(t2)) 
              for t0,t1,t2 in [("Lower","lower","n_lower"),
                               ("Upper","upper","n_upper")]]
        
        ax.text(1.03, 1, '\n'.join(tuple(s)), transform=ax.transAxes, 
                fontsize=12, va='top', ha='left')

    def __rescale__(self, x0:np.ndarray, 
                    x1:np.ndarray, x=None) -> np.ndarray:
    
        '''Rescale x0 to x1'''
        x = x0 if x is None else x
        norm = (x-min(x0))/(max(x0)-min(x0))
        try: return norm*(max(x1)-min(x1))+min(x1)
        except: return np.full(len(a), np.nan)
        
def __numfmt__(n:Union[int, float], 
               w:int=15, d:int=4) -> str:
    
    '''Apply number format'''
    if (isinstance(n, int) & (abs(n)<1e6)) | (n==0): 
        return "{:,.0f}".format(n)
    elif (isinstance(n, float) & (abs(n)<1e6)):
        return "{:,.4f}".format(n)
    else: return "{:,.4E}".format(n)
    
class Descriptive(UnivariateOutliers, DescStatsPlot):
    
    '''
    Generate descriptive statistics, which include those 
    that summarize the central tendency, dispersion and 
    shape of a dataset’s distribution, excluding NaN.
    
    Parameters
    ----------
    methods : list of str, default=None
        Method of capping outliers i.e. {"iqr", "mad", 
        "grubb", "mae", "sigma", "gesd", "pct"}. If None, 
        it defaults to all methods available, except "gesd". 
        See UnivariateOutliers.__doc__ for more details.
        
    plot_kwds : keywords
        Keyword arguments to be passed to kernel density 
        estimate plot <DescStatsPlot>.
    
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
                self.kwds = dict([(k,plot_kwds[k]) 
                                  for k in keys])
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
            Summary statistics of Dataframe provided.
            See Descriptive.__num__.__doc__ for more details.

        References
        ----------
        .. [1] https://www.itl.nist.gov/div898/handbook/eda/
               section3/eda35b.htm
               
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
            self.DescStatsPlot = \
            DescStatsPlot(X, self.num_info_, **self.kwds)
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
            - "min"        : 0th percentile (minimum)
            - "pct25"      : 25th percentile
            - "pct50"      : 50th percentile (median)
            - "pct75"      : 75th percentile
            - "max"        : 100th percentile (maximum)
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
                         ("unique" , np.unique(x).shape[0]), 
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
        
        self.num_info = pd.DataFrame(data)[num_fields]\
        .set_index('variable')\
        .rename(columns={"pct25":"25%",
                         "pct50":"50%",
                         "pct75":"75%",
                         "f_skewness":"fisher skew",
                         "g_skewness":"galton skew"})\
        .sort_index().T
    
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
        
    def plotting(self, var, ax=None, xlim=None, y=None):
        
        '''Using <DescStatsPlot> to plot univariate'''
        self.DescStatsPlot.plotting(var, ax ,xlim, y)
        plt.tight_layout()