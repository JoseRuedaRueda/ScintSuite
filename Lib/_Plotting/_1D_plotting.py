import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

__all__ = ['p1D_shaded_error', 'remove_lines', 'overplot_trace', 'multiline']


def p1D_shaded_error(ax, x, y, u_up, color='k', alpha=0.1, u_down=None,
                     line_param={}, line=True):
    """
    Plot confidence intervals

    Jose Rueda: jrrueda@us.es

    Basic shaded region between y + u_up and y - u_down. If no u_down is
    provided, u_down is taken as u_up

    :param  ax: Axes. The axes to draw to
    :param  x: The x data
    :param  y: The y data
    :param  u_up: The upper limit of the error to be plotted
    :param  u_down: (Optional) the bottom limit of the error to be plotter
    :param  color: (Optional) Color of the shaded region
    :param  alpha: (Optional) Transparency parameter (0: full transparency,
    1 opacity)
    :param  line: If true, the line x,y will be also plotted
    :param  line_param: (optional) Line parameters to plot the central line
    :return ax with the applied settings
    """
    if ax is None:
        fig, ax = plt.subplots()
    if u_down is None:
        u_down = u_up

    ax.fill_between(x, (y - u_down), (y + u_up), color=color, alpha=alpha,
                    label='__noname__')
    if line:
        if 'color' not in line_param:
            line_param['color'] = color
        ax.plot(x, y, **line_param)
    return ax


def remove_lines(ax):
    """
    Remove Strike Map from the plot

    Jose Rueda: jrrueda@us.es

    note: this is created to e used inside the GUI, although you can use it
    if you like. It will remove all lines (python line plots) from your
    axis

    :param  ax: axis to 'clean'

    :return : nothing, just 'clean the axis' from the StrikeMap
    """
    for i in range(len(ax.lines)):
        ax.lines[-1].remove()


def overplot_trace(ax, x, y, line_params={}, ymin=0., ymax=0.95):
    """
    Over plot a time traces over figure

    Jose Rueda Rueda: jrreda@us.es

    Notice, we will just plot the trace x,y on top of the current figure. y
    will be normalise such that its minimum is ymin * axis yscale and its max
    is ymax * axis y scale

    :param  ax: ax where to plot
    :param  x: x of the line to plot
    :param  y: y of the line to plot
    :param  line_params: dictionary for plt.plot with the line parameters
    :param  ymin: minimum normalization percentage
    :param  ymax: maximum normalization percentage
    """
    # --- Create the transformation
    # the x coords of this transformation are data, and the y coord are axes
    trans = \
        mpl.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
    # --- Normalise the y data
    ydummy = (y - y.min()) / (y.max() - y.min()) * (ymax - ymin) + ymin
    ax.plot(x, ydummy, transform=trans, **line_params)
    plt.draw()


def multiline(xs, ys, c, ax=None, line_params: dict = {},
              cmap='bwr'):
    """
    Plot lines with different colorings

    This plots a series of lines with given colors.

    Retun an object mapable for colorbars

    extracted from https://stackoverflow.com/questions/38208700/
    matplotlib-plot-lines-with-colors-through-colormap

    :param  xs: iterable container of x coordinates
    :param  ys: iterable container of y coordinates
    :param  c: iterable container of numbers mapped to colormap
    :param  ax (optional): Axes to plot on.
    :param  line_params: dict with the parameters for the line plotting
    Notes
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for line i

    param lc: LineCollection instance.
    """
    # find axes
    if ax is None:
        fig, ax = plt.subplots()

    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, cmap=cmap, **line_params)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array

    Taken from:
    https://stackoverflow.com/questions/8500700/how-to-plot-a-gradient-color-line-in-matplotlib
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def colorline(ax, x, y, z, **kwargs):
    """
    Plot lines coloured according to a given weight.

    Taken from:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html

    :param ax: axis to plot the data.
    :param x: values along the horizontal direction of the line to plot.
    :param y: values along the vertical direction of the line to plot.
    :param z: values of the weight color for each part of the line.
    :param kwargs: keyword arguments to send to axis.addCollection. Among others:
        :cmap: colormap for the plotting.
        :linewidth: width of the line.
        :linestyle: style of the line.
    ...
    """

    # Special case if a single number:
    try:
        _ = iter(z)
    except TypeError:
        z = np.array([z])

    z = np.asarray(z)

    vmin = z.min()
    vmax = z.max()

    options = { 'linestyle': '-',
                'alpha': 1.0,
                'linewidth': 1.0,
                'norm': plt.Normalize(vmin, vmax)
               }

    options.update(kwargs)


    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, **options)

    ax.add_collection(lc)

    return lc


