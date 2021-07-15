'''
Univariate-related functions:
[1] qq_plot
[2] chi2_test
[3] ks_test
[4] UnivariateOutliers
[5] MatchDist
[6] Descriptive
[7] DescStatsPlot
[8] BoxPlot
[9] Compare2samp

Authors: Danusorn Sitdhirasdr <danusorn.si@gmail.com>
versionadded:: 10-07-2021

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
from sklearn.preprocessing import OrdinalEncoder

__all__ = ["qq_plot", 
           "chi2_test", 
           "ks_test",
           "UnivariateOutliers" ,
           "MatchDist", 
           "Descriptive",
           "DescStatsPlot",
           "BoxPlot", 
           "Compare2samp", "column_dtype"]

def qq_plot(x, dist="norm", bins=10):
    
    '''
    Q–Q (quantile-quantile) plot is a probability plot, which 
    is a graphical method for comparing two distributions.
    
    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If `dist` is 
        a function, it must have an interface similar to 
        <scipy.stats._continuous_distns>.
    
    bins : int, default=10
        It defines the number of quantile bins between 1st and 
        99th quantiles.
    
    Returns
    -------
    QQ_plot : collections.namedtuple
        A tuple subclasses with named fields as follow:
        - r : float 
          Pearson’s correlation coefficient.
        - cv : float
          Critival value given (α, df)= (0.05, N-2).
        - rmse : float
          Root Mean Square Error, where error is defined as 
          difference between x and theoretical dist.
        - dist_name : str
          Name of cont. distribution function <scipy.stats>. 
        - params : tuple
          Tuple of output from <scipy.stats.rv_continuous.fit> i.e. 
          MLEs for shape (if applicable), location, and scale 
          parameters from data.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [2] https://www.researchgate.net/publication/291691147_
           A_modified_Q-Q_plot_for_large_sample_sizes
    
    Examples
    --------
    >>> from scipy import stats
    
    Create normal random variable x ~ N(μ,σ) = (2,2).
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, 
    ...                                    random_state=0)
    
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
        If `dist` is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If `dist` is a 
        function, it must have an interface similar to 
        <scipy.stats._continuous_distns>.
    
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
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, 
    ...                                    random_state=0)
    
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
    In the test of Chi-Square (χ2) for homogeneity of proportion, 
    the null hypothesis says that the distribution of sample data 
    fit a distribution from a certain population or not.

    .. versionadded:: 30-05-2021
    
    Parameters
    ----------
    x : array-like (1-dimensional) of float
        Input data.
    
    dist : str or function, default="norm"
        If `dist` is a string, it defines the name of continuous 
        distribution function under <scipy.stats>. If `dist` is a 
        function, it must have an interface similar to 
        <scipy.stats._continuous_distns>.
    
    bins : int or sequence of scalars, default=10
        If `bins` is an int, it defines the number of equal-sample 
        bins. If `bins` is a sequence, it defines a monotonically 
        increasing array of bin edges, including the rightmost edge.

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
    >>> x = stats.norm(loc=2, scale=2).rvs(size=500, 
    ...                                    random_state=0)
    
    H0 : data follows a normal distribution.
    Ha : data does not follow a normal distribution.
    
    >>> chi2_test(x, dist="norm")
    Chi2_Test(cv=0.4773224553586323, df=9, 
    ...       pvalue=0.9999750744566653, 
    ...       dist='norm', 
    ...       params=(1.9492911213351323, 1.9963135546858515))
    
    If α is 5% (0.05), we can not reject the null hypothesis 
    (0.99 > 0.05). Or we can determine the critical value as 
    follows:
    
    >>> df = 10 - 1
    >>> cv = chi2.ppf(0.95, df)
    16.9190
    
    We cannot reject the null hypotheis since χ2 is 0.4773, 
    which is less than χ2(α=5%, df=10-1) = 16.9190
    
    '''
    keys = ['cv', 'df', 'pvalue', 'dist', 'params']
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

    # Critical value and degrees of freedom.
    cv = ((observe-expect)**2/expect).sum()
    df = max(len(bins[1:])-1,1)
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
            An array of labels (int).
            
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
        kwargs = {"hist"  : dict(color="#d1d8e0", bins=bins, alpha=0.5), 
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
 
class Compare2samp:
    
    '''
    Compare two sets of sample by using Chi-Square test for 
    homogeneity [1,2] and Kolmogorov-Smirnov [3,4] tests.
    
    versionadded:: 10-07-2021
    
    Parameters
    ----------
    bins : int, default=10
        Number of Chi-Square bins to start off with.
    
    equal_width : bool, default=True
        If True, it uses equal-width binning, otherwise 
        equal-sample binning is used instead.
        
    max_category : int, default=100
        If number of unique elements from column with "object" 
        dtype, is less than or equal to max_category, its 
        dtype will be converted to "category". max_category 
        must be greater than or equal to 2.
    
    frac : float, default=0.01
        It defines a minimum fraction (%) of expected samples 
        per bin. A minimum number of samples resulted from
        frac is 5.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Goodness_of_fit
    .. [2] https://courses.lumenlearning.com/wmopen-concepts-
           statistics/chapter/test-of-homogeneity/
    .. [3] https://en.wikipedia.org/wiki/Kolmogorov–Smirnov_test
    .. [4] https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.stats.ks_2samp.html
    
    Attributes
    ----------
    result : collections.OrderedDict
        The order of keys is arranged according to input
        variable. Within each key, it contains "Stats" 
        (collections.namedtuple) with following fields:
        - "variable"    : Variable name
        - "chi2_cv"     : Chi-Square test critival value
        - "chi2_df"     : Chi-Square degrees of freedom
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_bins"   : Chi-Square bin edges
        - "ks_stat"     : Kolmogorov-Smirnov test critival value
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "dtype"       : Data type

    info : pd.DataFrame
        Information table is comprised of:
        - "variable"    : Variable name
        - "chi2_cv"     : Chi-Square test critival value
        - "chi2_df"     : Chi-Square degrees of freedom
        - "chi2_pvalue" : Chi-Square test p-value
        - "chi2_bins"   : Number of Chi-Square bins
        - "ks_stat"     : Kolmogorov-Smirnov test critival value
        - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
        - "dtype"       : Data type
    
    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> X, y = fetch_openml("titanic", version=1, 
    ...                     as_frame=True, return_X_y=True)
    
    Take a random sample of items from an axis of object.
    >>> random_X = X.sample(100).copy()
    
    Fit model
    >>> model = Compare2samp(bins=10, equal_width=True, 
    ...                      max_category=100).fit(X, random_X)
    
    Result
    >>> model.result
    
    Summary result
    >>> model.info
    
    '''
    def __init__(self, bins=10, equal_width=True, max_category=100, frac=0.01):
        
        self.bins = bins
        self.equal_width = equal_width
        self.max_category = max_category
        self.frac = min(np.fmax(frac, np.finfo("float32").eps), 0.9)
    
    def fit(self, X1, X2, use_X1=True):
        
        '''
        Fit model.
        
        Parameters
        ----------
        X1, X2 : array-like or pd.DataFrame
            Two DataFrames of sample observations, where their 
            sample sizes can be different but they must have
            the same number of features (columns).
            
        use_X1 : bool, default=True
            If True, it uses X1, and X2 as expected and observed
            samples, respectively, and vice versa when use_X1 is
            False.
    
        Attributes
        ----------
        result : collections.OrderedDict
            The order of keys is arranged according to input
            variable. Within each key, it contains "Stats" 
            (collections.namedtuple) with following fields:
            - "variable"    : Variable name
            - "chi2_cv"     : Chi-Square test critival value
            - "chi2_df"     : Chi-Square degrees of freedom
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_bins"   : Chi-Square bin edges
            - "ks_stat"     : Kolmogorov-Smirnov test critival value
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "dtype"       : Data type
     
        info : pd.DataFrame
            Information table is comprised of:
            - "variable"    : Variable name
            - "chi2_cv"     : Chi-Square test critival value
            - "chi2_df"     : Chi-Square degrees of freedom
            - "chi2_pvalue" : Chi-Square test p-value
            - "chi2_bins"   : Number of Chi-Square bins
            - "ks_stat"     : Kolmogorov-Smirnov test critival value
            - "ks_pvalue"   : Kolmogorov-Smirnov test p-value
            - "dtype"       : Data type
        
        '''
        # Convert `X` to pd.DataFrame
        X1 = _to_DataFrame(X1).copy()
        X2 = _to_DataFrame(X2).copy()
        
        # Assign to expected and observed samples
        x_exp, x_obs = (X1, X2) if use_X1 else (X2, X1)
        x_exp = column_dtype(x_exp, self.max_category)
        x_obs = column_dtype(x_obs, self.max_category)
        self.n_ = [x_exp.shape[0], x_obs.shape[0]]
        n_min = max(int(self.frac * x_exp.shape[0]), 5)
       
        # Numeric and catigorical features from expected observation
        cat_features = list(x_exp.select_dtypes(include="category"))
        num_features = list(x_exp.select_dtypes(include=np.number))
            
        # Initialize parameters.
        self.fields = ["chi2_cv", "chi2_df", "chi2_pvalue", 
                       "chi2_bins", "ks_stat", "ks_pvalue", "dtype"]
        Stats = collections.namedtuple('Stats', self.fields)   
        self.result = collections.OrderedDict()
        
        fields = ["f_exp", "f_obs", "type"]
        Params = collections.namedtuple('Params', fields)
        self.hist_data = collections.OrderedDict()

        # ===== Numerical Features =====
        for feat in num_features:
        
            data1 = x_exp[feat].values.copy()
            data2 = x_obs[feat].values.copy()

            # Calculate bin edges, given binning method.
            chi2_bins = self.__bins__(data1, self.bins, self.equal_width)
            chi2_bins = self.__leqx__(data1, chi2_bins, n_min)

            # Frequency of expected and observed samples
            f_exp = self.__freq__(data1, chi2_bins)
            f_obs = self.__freq__(data2, chi2_bins)
            f_exp = np.where(f_exp==0, np.finfo(float).eps, f_exp)

            # Chi-Square test for goodness of fit.
            chi2_cv = ((f_obs-f_exp)**2/f_exp).sum()*100
            chi2_df = max(len(chi2_bins)-2,1)
            chi2_pvalue = 1-stats.chi2.cdf(chi2_cv, df=chi2_df)

            # Kolmogorov-Smirnov test for goodness of fit.
            kwd = dict(alternative='two-sided', mode='auto')
            ks_stat, ks_pvalue = stats.ks_2samp(data1, data2, **kwd)
            
            self.hist_data[feat] = Params(*(f_exp, f_obs, "number"))
            self.result[feat] = Stats(*(chi2_cv, chi2_df, chi2_pvalue, 
                                        chi2_bins, ks_stat, ks_pvalue, 
                                        data1.dtype))
            
        # ===== Categorical Features =====
        # Keyword argument for OrdinalEncoder
        kwds = dict(categories='auto', dtype=np.int32, unknown_value=-1,
                    handle_unknown="use_encoded_value")
        
        # Fit and transform data.
        encoder = OrdinalEncoder(**kwds).fit(x_exp[cat_features])
        cat_exp = encoder.transform(x_exp[cat_features])
        cat_obs = encoder.transform(x_obs[cat_features])
        
        for n,feat in enumerate(cat_features):
            
            data1 = cat_exp[:,n].copy()
            data2 = cat_obs[:,n].copy()
            
            # Calculate bin edges, given binning method.
            chi2_bins = self.__bins__(data1, max(data1)+1, self.equal_width)
            chi2_bins = self.__leqx__(data1, chi2_bins, n_min)
            
            # Frequency of expected and observed samples
            f_exp = self.__freq__(data1, chi2_bins, len(chi2_bins))
            f_obs = self.__freq__(data2, chi2_bins, len(chi2_bins))
            f_exp = np.where(f_exp==0, np.finfo(float).eps, f_exp)
            
            # Chi-Square test for goodness of fit.
            chi2_cv = ((f_obs-f_exp)**2/f_exp).sum()*100
            chi2_df = max(len(chi2_bins)-2,1)
            chi2_pvalue = 1-stats.chi2.cdf(chi2_cv, df=chi2_df)
            
            # Change chi2_bins format 
            # i.e. (Group(x), [element(x,1), .., element(x,n)])
            index = np.digitize(np.arange(0, max(data1)+1), chi2_bins)
            chi2_bins = [(i,list(encoder.categories_[n][index==i])) 
                         for i in np.unique(index)]
            
            self.hist_data[feat] = Params(*(f_exp, f_obs, "category"))
            self.result[feat] = Stats(*(chi2_cv, chi2_df, chi2_pvalue, 
                                        chi2_bins, np.nan, np.nan, 
                                        "category"))
        
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
        Determine frequency (include np.nan). This function
        uses np.digitize() to determine the indices of the bins 
        to which each value in input array belongs. If any 
        values in x are less than the minimum bin edge bins[0], 
        they are indexed for bin 1. Whereas any values that are 
        greater than the maximum bin edge bin[-1] or are np.nan, 
        they are indexed for bin "len(bins)+1" or "max_index".
        
        '''
        if max_index is None: max_index = len(bins)+1
        indices = np.clip(np.digitize(x, bins), 1, max_index)
        return np.array([sum(indices==n) 
                         for n in range(1, max_index)])/len(x)
        
    def __bins__(self, x, bins, equal_width=True):
        
        '''
        According to binning method (equal-width or equal-sample),
        this function generates 1-dimensional and monotonic array 
        of bins. The last bin edge is the maximum value in x plus 
        np.finfo("float32").eps.
        
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
        To ensure that the sample size is appropriate for the use 
        of the test statistic, we need to ensure that frequency in 
        each bin must be greater than "n_min". Bin is collasped to 
        its immediate left bin if above condition is not met, except 
        the first bin.
        
        '''
        notnan = x[~np.isnan(x)]
        while True:
            leq5  = (np.histogram(notnan, bins)[0]<n_min)
            index = np.fmax(np.argmax(leq5),1)
            if sum(leq5)==0: return bins
            else: bins = np.delete(bins, index)

    def plotting(self, var, ax=None, colors=None, tight_layout=True, 
                 decimal=0, expect_kwds=None, observe_kwds=None, 
                 xticklabel_format=None, max_display=2):
        
        '''
        Plot Chi-Square Goodness of Fit Test.

        Parameters
        ----------
        var : str
            Variable name in self.info (attribute).

        ax : Matplotlib axis object, default=None
            Predefined Matplotlib axis. If None, `ax` is created 
            with default figsize.

        colors : list of color-hex, default=None
            Number of color-hex must be greater than 1. If None, 
            it uses default colors from Matplotlib.

        tight_layout : bool, default=True
            If True, it adjusts the padding between and around 
            subplots i.e. plt.tight_layout().
            
        decimal : int, default=0
            Decimal places for annotation of value(s).
            
        expect_kwds : keywords, default=None
            Keyword arguments of expected samples to be passed 
            to "ax.bar".
         
        observe_kwds : keywords, default=None
            Keyword arguments of observed samples to be passed 
            to "ax.bar".
        
        xticklabel_format : string formatter, default=None
            String formatters (function) for ax.xticklabels values. 
            If None, it defaults to "{:,.2f}".format.
        
        max_display : int, default=1
            Maximum number of categories to be displayed. This is
            available only when dtype=="category".

        Returns
        -------
        ax : Matplotlib axis object

        '''
        # Get values from self.hist_data
        f_exp = self.hist_data[var].f_exp
        f_obs = self.hist_data[var].f_obs
        dtype = self.hist_data[var].type
        data  = self.result[var]
        bins  = data.chi2_bins
        x = np.arange(len(f_exp))
        
        # x-ticklabels number format
        if xticklabel_format is None: n_format = "{:,.2f}".format
        else: n_format = xticklabel_format
                
        xticklabels = []
        if dtype=="number":
            
            for n in np.arange(len(f_exp)):
                if n < len(f_exp)-1: r = f"(<{n_format(bins[n+1])})"
                else: r = f"(≥{n_format(bins[-1])})" + "\nor missing"
                xticklabels.append(f"Group {n+1}" + "\n" + r)
    
        elif dtype=="category":
            
            for n,m in bins:
                if max_display>0:
                    # format = {"A","B",...(n)}
                    n_set = [f'"{s}"' for s in np.array(m)[:max_display]]
                    if len(m)>max_display: 
                        n_set.append("…({:,.0f})".format(len(m)))
                    xticklabels.append(f"Group {n}" + "\n{" + ",".join(n_set) + "}")
                else: xticklabels.append(f"Group {n}" + "\n({:,.0f})".format(len(m)) )

        # Create matplotlib.axes if ax is None.
        width = np.fmax(len(f_exp)*0.8,6)
        ax = self.__ax__(ax, (width, 4.3))

        # Get default line color.
        colors = self.__colors__(ax, colors, 2)
        
        num_format = ("{:,." + str(decimal) + "%}").format
        anno_kwds = dict(xytext =(0,4), textcoords='offset points', 
                         va='bottom', ha='center', fontsize=10, 
                         fontweight='demibold')
     
        # Vertical bar (Expect).
        kwds = dict(width=0.4, ec='k', alpha=0.9, color=colors[0], 
                    label='Expect (n={:,.0f})'.format(self.n_[0]))
        ax.bar(x-0.25, f_exp, **({**kwds, **expect_kwds} if 
                                 expect_kwds is not None else kwds))
        
        # Annotation (Expect).
        kwds = {**anno_kwds, **dict(color=colors[0])}
        for xy in zip(x-0.25, f_exp): 
            ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
            
        # Vertical bar (Observe).   
        kwds = dict(width=0.4, ec='k', alpha=0.9, color=colors[1], 
                    label='Observe (n={:,.0f})'.format(self.n_[1]))    
        ax.bar(x+0.25, f_obs, **({**kwds, **observe_kwds} if 
                                 observe_kwds is not None else kwds))
        
        # Annotation (Observe).
        kwds = {**anno_kwds, **dict(color=colors[1])}
        for xy in zip(x+0.25, f_obs): 
            ax.annotate(num_format(min(xy[1],1)), xy, **kwds)
            
        for spine in ["top", "left", "right"]:
            ax.spines[spine].set_visible(False)
        
        # Plot title.
        title  = "Variable : {}\n".format(var)  
        pvalue = lambda v : "N/A" if np.isnan(v) else "{:,.0%}".format(v)
        args   = (pvalue(data.chi2_pvalue), pvalue(data.ks_pvalue)) 
        title += r"p-value ($\chi^{2}$, $KS$) : " + "({}, {})".format(*args)
        ax.set_title(title, fontweight='demibold', fontsize=12)
        
        # Set labels.
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(-0.5, len(f_exp)-0.5)
        ax.set_yticks([])
        ax.set_yticklabels('')
        ax.set_ylim(0, max(max(f_exp), max(f_obs))/0.75)
        ax.legend(loc=0)
        if tight_layout: plt.tight_layout()
        return ax
    
    def __ax__(self, ax, figsize):
        
        '''Private: create axis if ax is None'''
        if ax is None: return plt.subplots(figsize=figsize)[1]
        else: return ax
        
    def __colors__(self, ax, colors, n=10):
        
        '''Private: get default line color'''
        if colors is not None: return colors
        else: return [ax._get_lines.get_next_color() 
                      for _ in range(n)]
                    
def column_dtype(X, max_category=100):
    
    '''
    This function converts columns to best possible dtypes,
    which are "float32", "int32" (boolean), "category", and
    "object". However, it ignores columns, whose dtype is 
    either np.datetime64 or np.timedelta64.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input array.
    
    max_category : int, default=100
        If number of unique elements from column with "object" 
        dtype, is less than or equal to max_category, its 
        dtype will be converted to "category". max_category 
        must be greater than or equal to 2.
    
    Returns
    -------
    Converted_X : pd.DataFrame
    
    '''
    # Select columns, whose dtype is neither 
    # datetimes, nor timedeltas.
    exclude = [np.datetime64, np.timedelta64] 
    columns = list(X.select_dtypes(exclude=exclude))
    
    if isinstance(max_category, int):
        max_category = max(2, max_category)
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
                if (int32-float32).sum()==0: 
                    Converted_X[var] = int32
                else: Converted_X[var] = float32
            else: Converted_X[var] = float32 
        except:
            objtype = x.astype("object")
            n_unq = len(objtype.unique())
            if n_unq<=max_category:
                Converted_X[var] = x.astype(str).astype("category") 
            else: Converted_X[var] = objtype
    return Converted_X