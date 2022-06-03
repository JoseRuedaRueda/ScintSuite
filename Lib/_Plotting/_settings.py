"""
Set the matplolib default parameters
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from Lib._Paths import Path
import f90nml
import logging
logger = logging.getLogger('ScintSuite.Plotting')
try:
    from cycler import cycler
except ImportError:
    text = "Not cycler module, default color of lines can't be changed"
    logger.warning('0: %s' % text)
paths = Path()

__all__ = ['plotSettings', 'axis_beauty']


# -----------------------------------------------------------------------------
# --- Plot settings
# -----------------------------------------------------------------------------
def plotSettings(plot_mode='software', usetex=False):
    """
    Set default options for matplotlib

    Anton J. van Vuuren ft. Jose Rueda

    @param plot_mode: set of options to load: software, article or presentation
    @param usetex: flag to use tex formating or not
    """
    # Load default plotting options
    filename = os.path.join(paths.ScintSuite, 'Data', 'MyData',
                            'plotting_default_param.cfg')
    nml = f90nml.read(filename)

    # Add font directories
    try:
        font_files = mpl.font_manager.findSystemFonts(fontpaths=paths.fonts)
        for font_file in font_files:
            mpl.font_manager.fontManager.addfont(font_file)
    except:
        logger.warning('5: No fonts founds. Using matplotlib default')

    # Set some matplotlib parameters
    mpl.rcParams["savefig.transparent"] = \
        nml['default']['transparent_background']

    mpl.rcParams['xtick.direction'] = nml['default']['tick_direction']
    mpl.rcParams['ytick.direction'] = nml['default']['tick_direction']

    mpl.rcParams['svg.fonttype'] = 'none'  # to edit fonts in inkscape

    # Try to set the fonttypes, only available in version > 3.5.2
    try:
        # for PDF backend
        plt.rcParams['pdf.fonttype'] = 42

        # for PS backend
        plt.rcParams['ps.fonttype'] = 42

        # for svg backend
        plt.rcParams['svg.fonttype'] = 'none'
    except:
        pass
    # Latex formating
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
    mpl.rcParams['legend.fontsize'] = nml[mode]['legend_font_size']
    mpl.rcParams['axes.titlesize'] = nml[mode]['title_font_size']
    mpl.rcParams['axes.labelsize'] = nml[mode]['axis_font_size']

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

    # Print and return
    print('Plotting options initialised')
    return


def axis_beauty(ax, param_dict: dict):
    """
    Modify axis labels, title, ....

    Jose Rueda: jrrueda@us.es

    @param ax: Axes. The axes to be modify
    @param param_dict: Dictionary with all the fields
    @return ax: Modified axis
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
