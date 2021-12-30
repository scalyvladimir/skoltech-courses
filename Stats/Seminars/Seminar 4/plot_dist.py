import numpy as np
import scipy
import scipy.stats
import matplotlib
import matplotlib as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

cmap = plt.get_cmap("tab10")

titlesize = 20
labelsize = 16
legendsize = labelsize
xticksize = 14
yticksize = xticksize

matplotlib.rcParams['legend.markerscale'] = 1.5     # the relative size of legend markers vs. original
matplotlib.rcParams['legend.handletextpad'] = 0.5
matplotlib.rcParams['legend.labelspacing'] = 0.4    # the vertical space between the legend entries in fraction of fontsize
matplotlib.rcParams['legend.borderpad'] = 0.5       # border whitespace in fontsize units
matplotlib.rcParams['font.size'] = 12

#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = 'Times New Roman'

rc = {"font.family" : "serif", 
      #"mathtext.fontset" : "stix",
      "mathtext.fontset" : "cm",
      "mathtext.rm": "serif"}

matplotlib.rcParams.update(rc)
matplotlib.rcParams["font.serif"] = ["Times New Roman"] + matplotlib.rcParams["font.serif"]

matplotlib.rcParams['axes.labelsize'] = labelsize
matplotlib.rcParams['axes.titlesize'] = titlesize

matplotlib.rcParams["axes.formatter.use_mathtext"] = True

matplotlib.rc('xtick', labelsize=xticksize)
matplotlib.rc('ytick', labelsize=yticksize)
matplotlib.rc('legend', fontsize=legendsize)

matplotlib.rc('font', **{'family':'serif'})
# matplotlib.rc('text', usetex=True)


def plot_empirical_pdf(x, y, ax, grid=True, color='k', border_ext=0.05,
                       xlim=None, ylim=None,
                       xlabel=None, ylabel=None, title=None, label=None,
                       arrowwidth=None, headwidth=3, headlength=5,
                       verbose=False, gen=None, gen_color="r--"):
    """
    Plot PDF of empirical (discrete) distribution.
    - ax:          This is the axis to plot data in.
    - grid:        If True, grid is plotted; default is True.
    - color:       Color for plotting arrows (delta-functions); default is black ('k').
    - border_ext:  Vertical and horizontal ranges will be expaned by (bordered_ext * 100)% of data ranges; 
        this parameter is not taken into account for axis X or Y if xlim or ylim are passed, respectively.
    - xlim:        Tuple (xmin, xmax); eliminates effect of border_ext.
    - ylim:        Tuple (ymin, ymax); eliminates effect of border_ext.
    - xlabel:      Lable for axis X.
    - ylabel:      Label for axis Y.
    - title:       title.
    - label:       Label to be shown in legend.
    - arrowwidth:  The absolute width of arrows used to represent delta functions of discrete PDF.
    - headwidth:   The relative width of arrows' heads; the default value is 3 absolute arrow widths.
    - headlength: 
    - verbose:     If True, prints debug information; default is False.
    """
    assert len(x) == len(y)
    x_width = np.max(x) - np.min(x) # ширина по оси X
    y_width = np.max(y)             # ширина по оси Y (минимальное значение - 0)
    if verbose:
        print('plot_point_masses: x_width = {}, y_width = {}, border_ext = {}'.format(x_width, y_width, border_ext))
    if arrowwidth is None:
        arrowwidth = 0.005 * y_width
    else:
        arrowwidth = arrowwidth * y_width
    X = x
    Y = np.zeros_like(X)
    U = np.zeros_like(X)
    V = y
    ax.quiver(X, Y, U, V, units='y', scale=1, scale_units='y', zorder=2,
              width=arrowwidth, headwidth=headwidth, headlength=headlength, color=color, label="Empirical PDF")
    if gen is not None:
        ax.plot(x, gen.pdf(x), gen_color, label="True PDF")
    x_low  = np.min(x) - border_ext * x_width
    x_high = np.max(x) + border_ext * x_width
    y_low = 0
    y_high = np.max(y) + border_ext * y_width
    # GRID
    if grid: ax.grid(which='both', linestyle='--', alpha=0.5)
    # LIMITS
    if xlim is None: xlim = (x_low, x_high)
    if ylim is None: ylim = (y_low, y_high)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if verbose:
        print('plot_point_masses: xlim is set to ({:.2f}, {:.2f})'.format(xlim[0], xlim[1]))
        print('plot_point_masses: ylim is set to ({:.2f}, {:.2f})'.format(ylim[0], ylim[1]))
    # LABELS
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title is not None:  ax.set_title(title)
    ax.legend()
    return ax


def plot_empirical_cdf(points, cumulatives, ax, grid=True, color='b', border_ext=0.05, 
                       xlim=None, ylim=None, xlabel=None, ylabel=None, title=None, label=None,
                       verbose=False, gen=None, gen_color="r--"):
    """
    Plot CDF of empirical (discrete) distribution.
    - points:      Coordinates of point masses
    - cumulatives: CDF values.
        It is assumed, that cumulatives[i] gives CDF value at intervals [points[i], points[i + 1]]
    - ax:          This is the axis to plot data in.
    - grid:        If True, grid is plotted; default is True.
    - color:       Color for plotting CDF; default is blue ('b').
    - border_ext:  Vertical and horizontal ranges will be expaned by (bordered_ext * 100)% of data ranges; 
        this parameter is not taken into account for axis X or Y if xlim or ylim are passed, respectively.
    - xlim:        Tuple (xmin, xmax); eliminates effect of border_ext.
    - ylim:        Tuple (ymin, ymax); eliminates effect of border_ext.
    - xlabel:      Lable for axis X.
    - ylabel:      Label for axis Y.
    - title:       title.
    - label:       Label to be shown in legend.
    - verbose:     If True, prints debug information; default is False.
    """
    y_low = 0; y_high = 1.0 + border_ext
    
    x_min = points[0]
    x_max = points[-1]
    x_width = x_max - x_min
    x_low  = x_min - border_ext * x_width
    x_high = x_max + border_ext * x_width

    ax.step(points, cumulatives, where='post', label="Empirical CDF", color=color, zorder=2)
    if gen is not None:
        ax.plot(points, gen.cdf(points), gen_color, label="True CDF")
    # GRID
    if grid: ax.grid(which='both', linestyle='--', alpha=0.5)
    # LIMITS
    if xlim is None: xlim = (x_low, x_high)
    if ylim is None: ylim = (y_low, y_high)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if verbose:
        print('plot_point_masses: xlim is set to ({:.2f}, {:.2f})'.format(xlim[0], xlim[1]))
        print('plot_point_masses: ylim is set to ({:.2f}, {:.2f})'.format(ylim[0], ylim[1]))
    # LABELS
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if title  is not None: ax.set_title(title)
    ax.legend()
    return ax 


def plot_est_hist_normal(error_gen, estimator, name, mu, sigma, sample_sizes, num_repeats, figsize=(20, 20), x_min=None, x_max=None, y_max=None):
    """
    In simple additive error model , try different sample sizes and check error distribution for normality.
    """
	
    y = np.linspace(x_min, x_max, 1000)
    k = len(sample_sizes)
    nrows = int(np.floor(np.sqrt(k)))
    ncols = int(np.ceil(k / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, (n, ax)  in enumerate(zip(sample_sizes, axes.flatten()[:k])):
        errors = error_gen(scale=sigma, size=(num_repeats, n))
        Y = mu + errors
        mY = estimator(Y)
        ax.hist(mY, bins=100, alpha=0.2, label=f'{name} on {n} samples', density=True, color=cmap(i))
        fy = scipy.stats.norm(loc=mY.mean(), scale=mY.std()).pdf(y)
        #print(f"loc={mY.mean()}, scale={mY.std()}")
        ax.plot(y, fy, label=f'Normal dist on {n} samples', color=cmap(i))
        ax.legend()
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
    

def plot_uncertainty(x_train, y_train, x_test, y_test, y_pred, y_lower, y_upper, title='Model', x_label=None, y_label=None, label1='', label2='', color=None, alpha=0.2):
    if type(color) is int:
        color = cmap(color)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    ax1.set_title(title)

    ax1.plot(x_test, y_test, 'k--', label='Theoretical')

    ax1.plot(x_train, y_train, 'o', label='Observed')

    ax1.plot(x_test, y_pred, 'g--', label='Prediction')
    ax1.fill_between(x_test, y_lower, y_upper, alpha=alpha, color=color, label=label1)
    ax1.set_xlabel(x_label)
    ax1.set_xlabel(y_label)

    ax1.legend()

    ax2.set_title("Width of the interval")
    ax2.fill_between(x_test, y_upper - y_pred, 0, alpha=alpha, color=color, label=label2)
    ax2.plot(x_train, np.zeros_like(x_train), 'o', color=cmap(0), label='Observations')
    ax2.set_xlabel(x_label)
    ax2.legend()
