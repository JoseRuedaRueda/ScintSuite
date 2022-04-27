"""Class to interact with the FBM"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Lib.SimulationCodes.FIDASIM._read import read_fbm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class FBM:
    """
    FIDASIM fast-ion distribution function

    Jose Rueda: jrrueda@us.es

    Note: only fulle tested with axi-symmetric distributions

    Public methods:
        - plotRZ: Plot the FI density (R,Z)
        - plotEPatRZ: Plot the FI distribution on E, pitch at a given R, Z
        - plotInt: integrate the FI distribution along 3 ranges in 3 of the
            variables and plot the results vs the 4th
    """

    def __init__(self, filename: str, verbose: bool = True):
        """
        Initalise the class

        @param filename: filename to be read
        @param verbose: bollean flag to print some info in the console
        """
        if not os.path.isfile(filename):
            raise Exception('File not found')
        self._dat = read_fbm(filename)

        # Print the information
        if verbose:
            print('TRANSP file: %s' % self._dat['cdf_file'])
            print('t: %f s' % self._dat['cdf_time'])
            print('Gyrocenter distribution: %i' % self._dat['fbm_gc'])
            print('Particle mass: %f amu' % self._dat['afbm'])

    def plotRZ(self, ax=None, cmap=None, IncludeColorbar: bool = True,
               scale: str = 'sqrt', units: str = 'm',
               interpolation: str = 'bicubic'):
        """
        Plot the FBM in RZ

        Basically plot the FBM density (which is the 4D FBM integrated in E and
        pitch)

        @param ax: axes where to plot, is None, new will be created
        @param cmap: if none, plasma will be used
        @param IncludeColorbar: bolean flag to include a colormap
        @param scale: log, sqrt or linear scale to plot
        @param units: units for R and z
        @param interpolation: interpolation methid to show the image
        """
        # --- Create the axis, if needed
        created = False
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        # --- Get the colormap, if needed
        if cmap is None:
            cmap = plt.get_cmap('plasma')
        # --- Prepare the scale to plot
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(clip=True)}
        else:
            extra_options = {}
        # --- Prepare the scale of the axis
        factors = {
            'm': 0.01,
            'cm': 1.00,
            'inch': 1.0/2.54,
        }
        factor = factors[units]
        # --- Plot the distribution
        img = ax.imshow(self._dat['denf'].T,
                        extent=[self._dat['rgrid'][0] * factor,
                                self._dat['rgrid'][-1] * factor,
                                self._dat['zgrid'][0] * factor,
                                self._dat['zgrid'][-1] * factor],
                        origin='lower',
                        interpolation=interpolation, cmap=cmap,
                        **extra_options)
        # --- If we created the axis, put some labels
        if created:
            ax.set_xlabel('R [%s]' % units)
            ax.set_xlabel('z [%s]' % units)
        # --- Include the colorbar
        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, label='FI density [a.u.]', cax=cax)

    def plotEPatRZ(self, R, z, ax=None, cmap=None,
                   IncludeColorbar: bool = True,
                   scale: str = 'sqrt', units: str = 'm',
                   interpolation: str = 'bicubic'):
        """
        Plot the distribution as function of E, pitch at an E,Z

        Note: By default R,z are expected in m! if you want to use cm or have
        fun with inches, use the 'units' input variable to change the scale

        @param R: r point where to plot
        @param z: z points where to plot
        @param ax: axes where to plot, is None, new will be created
        @param cmap: if none, plasma will be used
        @param IncludeColorbar: bolean flag to include a colormap
        @param scale: log, sqrt or linear scale to plot
        @param units: units for R and z
        @param interpolation: interpolation methid to show the image
        """
        # --- Create the axis, if needed
        created = False
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        # --- Get the colormap, if needed
        if cmap is None:
            cmap = plt.get_cmap('plasma')
        # --- Prepare the scale to plot
        if scale == 'sqrt':
            extra_options = {'norm': colors.PowerNorm(0.5)}
        elif scale == 'log':
            extra_options = {'norm': colors.LogNorm(clip=True)}
        else:
            extra_options = {}
        # --- Prepare the scale of the axis
        factors = {
            'm': 100.0,
            'cm': 1.00,
            'inch': 2.54,
        }
        factor = factors[units]
        # --- find the cell
        ir = np.argmin(np.abs(self._dat['rgrid'] - R * factor))
        iz = np.argmin(np.abs(self._dat['zgrid'] - z * factor))
        # --- Plot the distribution
        img = ax.imshow(self._dat['fbm'][..., ir, iz].squeeze().T,
                        extent=[self._dat['energy'][0],
                                self._dat['energy'][-1],
                                self._dat['pitch'][0],
                                self._dat['pitch'][-1]],
                        origin='lower', aspect='auto',
                        interpolation=interpolation, cmap=cmap,
                        **extra_options)
        # include a small note with the position
        ax.text(0.95, 0.9, 'R = %.2f %s, z = %.2f %s' %
                (self._dat['rgrid'][ir] / factor, units,
                 self._dat['zgrid'][iz] / factor, units),
                horizontalalignment='right',
                color='w', verticalalignment='bottom',
                transform=ax.transAxes)
        # --- If we created the axis, put some labels
        if created:
            ax.set_xlabel('Energy [keV]')
            ax.set_ylabel('Pitch []')
        # --- Include the colorbar
        if IncludeColorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, label='FI density [a.u.]', cax=cax)
        return ax

    def plotInt(self, R: tuple = None, z: tuple = None, E: tuple = None,
                P: tuple = None, ax=None, line_params: dict = {},
                verbose: bool = True):
        """
        Plot the 1D results of performing a 3D integration

        Note, 3 and only 3 intervals should be given

        @param R: Tuple containing (Rmin, Rmax) for the integration
        @param z: Tuple containing (zmin, zmax) for the integration
        @param E: Tuple containing (Emin, Emax) for the integration
        @param P: Tuple containing (Pmin, Pmax) for the integration
        @param line_params: dictionary with line parameters for matplotlib
        @param verbose: if True, print some basic infor in the console
        """
        possible = {
            'R': 'R',
            'E': 'E',
            'z': 'z',
            'P': '$\\lambda'
        }
        counter = 0
        if R is None:
            var = 'R'
            counter += 1
            axis_to_integrate = (0, 1, 3)
            R = (self._dat['rgrid'][0], self._dat['rgrid'][-1])
            x = self._dat['rgrid']
        if z is None:
            var = 'z'
            counter += 1
            axis_to_integrate = (0, 1, 2)
            z = (self._dat['zgrid'][0], self._dat['zgrid'][-1])
            x = self._dat['zgrid']
        if E is None:
            var = 'E'
            counter += 1
            axis_to_integrate = (1, 2, 3)
            E = (self._dat['energy'][0], self._dat['energy'][-1])
            x = self._dat['energy']
        if P is None:
            var = 'P'
            counter += 1
            axis_to_integrate = (0, 2, 3)
            P = (self._dat['pitch'][0], self._dat['pitch'][-1])
            x = self._dat['pitch']

        if counter != 1:
            print('Only one integration limit should be None')

        # Get the inteval in R
        ir0 = np.argmin(np.abs(self._dat['rgrid'] - R[0]))
        ir1 = np.argmin(np.abs(self._dat['rgrid'] - R[1]))
        ir1 = max(ir0 + 1, ir1 + 1)
        rmin = self._dat['rgrid'][ir0]
        rmax = self._dat['rgrid'][ir1 - 1]
        # Get the inteval in z
        iz0 = np.argmin(np.abs(self._dat['zgrid'] - z[0]))
        iz1 = np.argmin(np.abs(self._dat['zgrid'] - z[1]))
        iz1 = max(iz0 + 1, iz1 + 1)
        zmin = self._dat['zgrid'][iz0]
        zmax = self._dat['zgrid'][iz1 - 1]
        # Get the interval in E
        ie0 = np.argmin(np.abs(self._dat['energy'] - E[0]))
        ie1 = np.argmin(np.abs(self._dat['energy'] - E[1]))
        ie1 = max(ie0 + 1, ie1 + 1)
        emin = self._dat['energy'][ie0]
        emax = self._dat['energy'][ie1 - 1]
        # Get the inteval in pitch
        ip0 = np.argmin(np.abs(self._dat['pitch'] - P[0]))
        ip1 = np.argmin(np.abs(self._dat['pitch'] - P[1]))
        ip1 = max(ip0 + 1, ip1 + 1)
        pmin = self._dat['pitch'][ip0]
        pmax = self._dat['pitch'][ip1 - 1]
        # integrate the distribution
        y = self._dat['fbm'][ie0:ie1, ip0:ip1, ir0:ir1, iz0:iz1].sum(
            axis=axis_to_integrate)

        # plot the distribution
        created = False
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y, **line_params)
        if created:
            ax.set_ylabel('FI distribution [a.u.]')
            ax.set_xlabel(possible[var])
        if verbose:
            print('E: %.2f - %.2f' % (emin, emax))
            print('p: %.2f - %.2f' % (pmin, pmax))
            print('R: %.2f - %.2f' % (rmin, rmax))
            print('z: %.2f - %.2f' % (zmin, zmax))
        return x, y
