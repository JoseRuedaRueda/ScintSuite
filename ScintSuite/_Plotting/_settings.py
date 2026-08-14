"""
Set the matplolib default parameters
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from ScintSuite._Paths import Path
import yaml

import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
import numpy as np


import logging
logger = logging.getLogger('ScintSuite.Plotting')
try:
    from cycler import cycler
except ImportError:
    text = "Not cycler module, default color of lines can't be changed"
    logger.warning('10: %s' % text)
paths = Path()

__all__ = ['plotSettings', 'axis_beauty']


# -----------------------------------------------------------------------------
# --- Plot settings
# -----------------------------------------------------------------------------
def plotSettings(plot_mode='software', usetex=False):
    """
    Set default options for matplotlib

    Anton J. van Vuuren ft. Jose Rueda

    :param  plot_mode: set of options to load: software, article or presentation
    :param  usetex: flag to use tex formating or not
    """
    # Load default plotting options
    filename = os.path.join(paths.ScintSuite, 'Settings.yml')
    with open(filename, 'r') as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception('Error reading the settings file')
    nml = settings['UserPlotStyles']

    # Add font directories
    try:
        font_files = mpl.font_manager.findSystemFonts(fontpaths=paths.fonts)
        for font_file in font_files:
            mpl.font_manager.fontManager.addfont(font_file)
    except:
        logger.warning('15: No fonts founds. Using matplotlib default')

    # Set some matplotlib parameters
    mpl.rcParams["savefig.transparent"] = \
        nml['default']['transparent_background']

    mpl.rcParams['xtick.direction'] = nml['default']['tick_direction']
    mpl.rcParams['ytick.direction'] = nml['default']['tick_direction']

    mpl.rcParams['svg.fonttype'] = 'none'  # to edit fonts in inkscape

    # mpl.rcParams["backend"] = 'Qt5Agg'
    # Try to set the font-types, only available in version > 3.5.2
    try:
        # for PDF backend
        plt.rcParams['pdf.fonttype'] = 42

        # for PS backend
        plt.rcParams['ps.fonttype'] = 42

        # for svg backend
        plt.rcParams['svg.fonttype'] = 'none'
    except:
        pass
    # Latex formatting
    mpl.rc('text', usetex=usetex)

    # Default plotting color
    try:
        mpl.rcParams['axes.prop_cycle'] = \
            cycler(color=nml['default']['default_line_colors'])
    except NameError:
        print("Not cycler module, default color of lines can't be changed")

    # from: https://stackoverflow.com/questions/21321670/
    #   how-to-change-fonts-in-matplotlib-python
    # https://www.w3schools.com/css/css_font.asp

    mode = plot_mode.lower()
    opt = {
        'family': nml[mode]['font_family'],
        'serif': [nml[mode]['font_name']],
        'size': nml[mode]['axis_font_size']
    }
    mpl.rc('font', **opt)
    try:
        mpl.rcParams['font.size'] = nml[mode]['inside_text_font_size']
    except KeyError:
        mpl.rcParams['font.size'] = nml[mode]['axis_font_size']

    mpl.rcParams['axes.titlesize'] = nml[mode]['title_font_size']
    plt.rcParams['figure.titlesize'] = nml[mode]['title_font_size']
    mpl.rcParams['axes.labelsize'] = nml[mode]['axis_font_size']
    mpl.rcParams['xtick.labelsize'] = nml[mode]['tick_font_size']
    mpl.rcParams['ytick.labelsize'] = nml[mode]['tick_font_size']
    mpl.rcParams['legend.fontsize'] = nml[mode]['legend_font_size']

    mpl.rcParams['lines.linewidth'] = nml[mode]['line_width']
    mpl.rcParams['lines.markersize'] = nml[mode]['marker_size']

    mpl.rcParams['xtick.major.size'] = nml[mode]['Major_tick_length']
    mpl.rcParams['xtick.major.width'] = nml[mode]['Major_tick_width']
    mpl.rcParams['xtick.minor.size'] = nml[mode]['minor_tick_length']
    mpl.rcParams['xtick.minor.width'] = nml[mode]['minor_tick_width']
    mpl.rcParams['ytick.major.size'] = nml[mode]['Major_tick_length']
    mpl.rcParams['ytick.major.width'] = nml[mode]['Major_tick_width']
    mpl.rcParams['ytick.minor.size'] = nml[mode]['minor_tick_length']
    mpl.rcParams['ytick.minor.width'] = nml[mode]['minor_tick_width']
    try:
        mpl.rcParams['ytick.direction'] = nml[mode]['ytick_direction']
        mpl.rcParams['xtick.direction'] = nml[mode]['xtick_direction']
    except KeyError:
        mpl.rcParams['ytick.direction'] = nml['default']['tick_direction']
        mpl.rcParams['xtick.direction'] = nml['default']['tick_direction']


    # Print and return
    logger.info('Plotting options initialised')
    return


def axis_beauty(ax, param_dict: dict):
    """
    Modify axis labels, title, ....

    Jose Rueda: jrrueda@us.es

    :param  ax: Axes. The axes to be modify
    :param  param_dict: Dictionary with all the fields
    :return ax: Modified axis
    """
    # Define fonts
    font = {}
    if 'fontname' in param_dict:
        font['fontname'] = param_dict['fontname']
    if 'fontsize' in param_dict:
        font['size'] = param_dict['fontsize']
        labelsize = param_dict['fontsize']
        # ax.tick_params(labelsize=param_dict['fontsize'])
    if 'xlabel' in param_dict:
        ax.set_xlabel(param_dict['xlabel'], **font)
    if 'ylabel' in param_dict:
        ax.set_ylabel(param_dict['ylabel'], **font)
    if 'yscale' in param_dict:
        ax.set_yscale(param_dict['yscale'])
    if 'xscale' in param_dict:
        ax.set_xscale(param_dict['xscale'])
    if 'tickformat' in param_dict:
        ax.ticklabel_format(style=param_dict['tickformat'], scilimits=(-2, 2),
                            useMathText=True)
        if 'fontsize' in param_dict:
            ax.yaxis.offsetText.set_fontsize(param_dict['fontsize'])
        if 'fontname' in param_dict:
            ax.yaxis.offsetText.set_fontname(param_dict['fontname'])
    if 'grid' in param_dict:
        if param_dict['grid'] is not None:
            if param_dict['grid'] == 'both':
                ax.grid(True, which='minor', linestyle=':')
                ax.minorticks_on()
                ax.grid(True, which='major')
            else:
                ax.grid(True, which=param_dict['grid'])
    if 'ratio' in param_dict:
        ax.axis(param_dict['ratio'])
    # Arrange ticks a ticks labels
    if 'fontsize' in param_dict:
        ax.tick_params(which='both', direction='in', color='k', bottom=True,
                       top=True, left=True, right=True, labelsize=labelsize)
    else:
        ax.tick_params(which='both', direction='in', color='k', bottom=True,
                       top=True, left=True, right=True)
    return ax


# -----------------------------------------------------------------------------
# %% Add the sqrt scaling for the axis
# -----------------------------------------------------------------------------
class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    
    Taken from: https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python
    
    Example usage:
    >>> import matplotlib.pyplot as plt
    >>> import ScintSuite as ss # This add the sqrt scale to matplotlib
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.sqrt(np.arange(100)), np.arange(100))
    >>> ax.set_xscale('sqrt')
    >>> plt.show()
    """
 
    name = 'sqrt'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform_non_affine(self, a): 
            return np.sign(np.array(a)) * np.abs(np.array(a))**0.5
 
        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()
 
    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
 
        def transform(self, a):
            return np.sign(np.array(a)) * np.abs(np.array(a))**2
 
        def inverted(self):
            return SquareRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.SquareRootTransform()
 
mscale.register_scale(SquareRootScale)