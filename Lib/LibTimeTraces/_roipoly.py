"""
RoiPoly

This module allows to select a polygonal region on a matplotlib plot.

Jose Rueda Rueda: jrrueda@us.es

Notice the package roipoly is really buggy in Spyder IDE, which causes several
problems in the suite, so we needed to create our own
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mplPath


class roipoly:
    """
    Create ROI with arbitrary shape from an image or via an input path provided
    by the user.

    Jose Rueda Rueda: jrrueda@us.es
    """

    def __init__(self, fig=None, ax=None, drawROI: bool = False,
                 path: float=None):
        """
        Initialise the ROI object either from an image-chosen ROI or with a
        precomputed ROI.

        @param fig: Figure where the desired image is plotted
        @param ax: axis where the image is plotted
        @param drawROI: flag to draw the ROI contour after selecting it
        @param path: input path with shape (N, 2) corresponding to (X, Y) path
        to compute the ROI.

        Notice, as a side effect, it will make ax to be the current axis, so be
        carefull if you use this function inside some other codeflow and you
        are plottings things elsewhere
        """
        # If the user provided a ROI via an input path, we can just skip
        # the image opening.
        if path is not None:
            path = np.atleast_2d(path)
            if path.ndim != 2:
                raise ValueError('The input path must correspond to'+\
                                 ' a collection (X, Y) points')
            if path.shape[1] != 2:
                raise ValueError('The input path must have axis=1 with size'+\
                                 ' = 2 (X, Y)')

            self.xpoints = path[:, 0]
            self.ypoints = path[:, 1]
        else:
            # Get the axis, if needed
            if fig is None:
                fig = plt.gcf()
            if ax is None:
                ax = plt.gca()
            self.fig = fig
            self.ax = ax
            # Show the figure and make the axis to be the current ones
            fig.show()
            plt.sca(ax)
            # Print instruction for the user
            print('Please select the vertex of the roi in the figure')
            print('Select each vertex with left click')
            print('Undo your selection with right click')
            print('Once you finished, right the middle button')
            print('You have 216 seconds')
            # Get the points
            dummy = np.array(plt.ginput(-1, timeout=216))
            self.xpoints = dummy[:, 0]
            self.ypoints = dummy[:, 1]

        # Draw if needed
        if drawROI:
            x = np.concatenate((self.xpoints, np.array([self.xpoints[0]])))
            y = np.concatenate((self.ypoints, np.array([self.ypoints[0]])))
            plt.plot(x, y)

    def getMask(self, currentImage):
        """
        Get the binary mask from the selected ROI points

        Taken from the toupy code https://github.com/jcesardasilva/toupy

        @param currentImage: Image (matrix) for which we want the ROI
        """
        ny, nx = np.shape(currentImage)
        poly_verts = [(self.xpoints[0], self.ypoints[0])]
        for i in range(len(self.xpoints) - 1, -1, -1):
            poly_verts.append((self.xpoints[i], self.ypoints[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        ROIpath = mplPath.Path(poly_verts)
        mask = ROIpath.contains_points(points).reshape((ny, nx))
        return mask
