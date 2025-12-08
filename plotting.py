
import colorcet as cc  # pip install colorcet
import matplotlib as mpl
from matplotlib import colors, rcParams, cm
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm, to_rgb
from matplotlib.axes import Axes
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, AutoMinorLocator, MaxNLocator
import matplotlib.pyplot as plt  # pip install matplotlib==3.8.4
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec


fs = 10
dpi = 200
config_figure = {'figure.figsize': (3, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],  # Avenir LT Std, Helvetica Neue LT Pro, Helvetica LT Std
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

