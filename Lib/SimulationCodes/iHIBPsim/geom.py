"""
Lib HIBP Geometry

This library contains the basic information to plot the injection geometry of
the i-HIBP beam.

Pablo Oyola - pablo.oyola@ipp.mpg.de
"""

import numpy as np
import matplotlib.pyplot as plt
import Lib
import random
from typing import Union
import Lib.LibData.AUG.DiagParam as libparms
import Lib.SimulationCodes.iHIBPsim as libhipsim
import Lib.SimulationCodes.Common.geometry as common_geom
import warnings
from copy import deepcopy
from Lib.LibUtilities import find_2D_intersection

BEAM_INF_SMALL = 0
BEAM_ANGULAR_DIVERGENCY = 1
BEAM_ENERGY_DISPERSION = 2
BEAM_FULLWIDTH = 3

# ----------------------------------------------------------------------------
# --- Beam geometry modules.
# ----------------------------------------------------------------------------

def R2beam(u: float, origin: float, R: float):
    """
    For a given beam trajectory in the (X, Y) described by the vector pair
    (ux, uy), the routine computes the beam coordinate associated to some
    major radius.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param u: beam direction vector in cartesian coordiantes.
    @param origin: cartesian coordinates of the origin.
    @param R:  major radius to transform to beam coordinate.
    """

    # Transforming the input into a 1D array.
    R = np.atleast_1d(R)

    # Radius of the origin.
    Rorigin = np.sqrt(origin[0]**2.0 + origin[1]**2.0)

    # CComputing the origin*direction vector:
    o_times_u = np.dot(u, origin)

    # Computing the discriminant:
    disc = o_times_u**2.0 + (R**2.0 - Rorigin**2.0)

    flags_nan = disc < 0
    disc[flags_nan] = 0.0

    s0 = - o_times_u + np.sqrt(disc)
    s1 = - o_times_u - np.sqrt(disc)

    # For points behing the injection point, we need a tweak:
    flags = R > Rorigin

    s_out = np.zeros(s0.shape)
    s_out[flags] = np.maximum(s0[flags], s1[flags])
    s_out[~flags] = np.minimum(s0[~flags], s1[~flags])

    return s_out


def generateBeamTrajectory(start: float, beta: float, theta: float=0.0,
                           Rmin: float=1.03, Rmax: float=None, Ns: int = 128):
    """
    Provided the port origin, this routine will generate the beam line
    (assuming it infinitely small) that would follow into the plasma.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param start: vector with 3 components in cartesian coordinates indicating
    the origin of the beam.
    @param beta: toroidal tilting angle, defined as the tilting wrt to the
    radial direction whose origin is in start.
    @param theta: tilting angle in the poloidal direction.
    @param Rmin: Minimum major radius. Set to the geometrical center of AUG.
    @param Rmax: Maximum major radius. If None, it will be chosen to be the
    Rmajor corresponding to the starting.
    @param Ns: number of points along the beam to compute.
    """

    beta = beta/180.0*np.pi
    theta = theta/180.0*np.pi

    if abs(beta) > np.pi/2:
        raise Exception('Toroidal tilting angle cannot be larger than pi/2')

    if abs(theta) > np.pi/2:
        raise Exception('Poloidal tilting angle cannot be larger than pi/2')

    # --- Getting the cylindrical coordinates of the beam injection point.
    Rstart = np.sqrt(start[0]**2.0 + start[1]**2.0)
    phistart = np.arctan2(start[1], start[0])

    if Rmax is None:
        Rmax = Rstart

    phiinj = phistart - beta

    u_inj = [ - np.cos(phiinj)*np.cos(theta),
              - np.sin(phiinj)*np.cos(theta),
                np.sin(theta) ]

    u_inj = u_inj/np.linalg.norm(u_inj)

    # --- Getting the Rmax and Rmin in beam coordinates.
    aux = R2beam(u=u_inj, origin=start, R=(Rmin,Rmax))
    smin = aux.min()
    smax = aux.max()

    sbeam = np.linspace(smin, smax, Ns)

    sbeam2, u_inj2 = np.meshgrid(sbeam, u_inj)

    beam_data = np.tile(start, (sbeam.size, 1)).T + u_inj2*sbeam2

    output = { 's_beam': sbeam,
               'u_inj': u_inj,
               'beam_cart': beam_data,
               'Rbeam': np.sqrt(beam_data[0, :]**2.0 + beam_data[1, :]**2.0),
               'zbeam': beam_data[2, :],
               'beta': beta,
               'theta': theta,
               'phiinj': phiinj,
               'thetainj': theta,
               'origin': start
             }

    return output


def plotBeam_poloidal(beam_data: dict, ax=None, fig=None,
                      pol_up: dict={}, pol_down: dict={}, plotDiv: bool=False,
                      drawvessel: bool=True, line_opts: dict={},
                      ax_options: dict={}):
    """
    Projects onto the poloidal plane the beam line.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param pol_up: same dictionary type as beam data, but for the upper limit
    (in poloidal) of the beam.
    @param pol_down: same dictionary type as beam data, but for the lower limit
    (in poloidal) of the beam.
    @param plotDiv: flag to plot the divergency of the beam.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot
    @param ax_options: options to decorate the axis.
    Send down to Lib.plt.axis_beauty.
    """

    ax_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1)
        ax_was_none = True

    if 'linewidth' not in line_opts:
        line_opts['linewidth'] = 2.0

    line_hndl = ax.plot(beam_data['Rbeam'], beam_data['zbeam'], '-',
                       **line_opts)

    if plotDiv:
        color = line_hndl[-1]._color
        div_hndl = ax.fill_between(pol_up['Rbeam'],
                                   pol_up['zbeam'], pol_down['zbeam'],
                                   linewidth=0.0, color=color, alpha=0.1)
    else:
        div_hndl = None


    if drawvessel:
        ax=Lib.plt.plot_vessel(projection='pol', units='m',
                               color='k', linewidth=0.75, ax=ax)

    if ax_was_none:
        ax_options['ratio'] = 'equal'
        ax_options['xlabel'] = 'Major radius R [m]'
        ax_options['ylabel'] = 'Height z [m]'
        ax_options['grid'] = 'both'

        ax = Lib.plt.axis_beauty(ax, ax_options)

    return ax, line_hndl, div_hndl


def plotBeam_toroidal(beam_data: dict, ax=None, fig=None,
                      tor_up: dict={}, tor_down: dict={}, plotDiv: bool=False,
                      drawvessel: bool=True, line_opts: dict={},
                      ax_options: dict={}):
    """
    Projects onto the toroidal plane the beam line.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param tor_up: same dictionary type as beam data, but for the upper limit
    (in toroidal) of the beam.
    @param tor_down: same dictionary type as beam data, but for the lower limit
    (in toroidal) of the beam.
    @param plotDiv: flag to plot the divergency of the beam.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot
    @param ax_options: options to decorate the axis.
    Send down to Lib.plt.axis_beauty.
    """

    ax_was_none = False
    if ax is None:
        fig, ax = plt.subplots(1)
        ax_was_none = True

    if 'linewidth' not in line_opts:
        line_opts['linewidth'] = 2.0

    line_hndl = ax.plot(beam_data['beam_cart'][0, :],
                        beam_data['beam_cart'][1, :], '-',
                        **line_opts)
    if plotDiv:
        color = line_hndl[-1]._color
        x = (tor_up['beam_cart'][0, 0],    tor_up['beam_cart'][0, -1],
             tor_down['beam_cart'][0, -1], tor_down['beam_cart'][0, 0])
        y = (tor_up['beam_cart'][1, 0],    tor_up['beam_cart'][1, -1],
             tor_down['beam_cart'][1, -1], tor_down['beam_cart'][1, 0])

        div_hndl = ax.fill(x, y, color=color, alpha=0.1, linewidth=0.0)
    else:
        div_hndl = None

    if drawvessel:
        ax=Lib.plt.plot_vessel(projection='tor', units='m',
                               color='k', linewidth=0.75, ax=ax)

    if ax_was_none:
        ax_options['ratio'] = 'equal'
        ax_options['xlabel'] = 'X [m]'
        ax_options['ylabel'] = 'Y [m]'
        ax_options['grid'] = 'both'

        ax = Lib.plt.axis_beauty(ax, ax_options)

    return ax, line_hndl, div_hndl


def plotBeam_3D(beam_data: dict, ax=None, fig=None,
                drawvessel: bool=True, diverg: dict= {}, line_opts: dict={},
                ax_options: dict={}, params3d: dict={}):
    """
    Plots the 3D vessel and the beam line in 3D.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param diverg: dictionary containing the data to plot the divergence.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot
    @param ax_options: options to decorate the axis.
    Send down to Lib.plt.axis_beauty.
    @param params3d: parameters to plot 3D surfaces.
    """

    if not 'plotDiv' in diverg:
        diverg['plotDiv'] = False

    ax_was_none = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax_was_none = True

    if 'linewidth' not in line_opts:
        line_opts['linewidth'] = 2.0

    if 'alpha' not in params3d:
        params3d['alpha'] = 0.25

    if 'phi_min' not in params3d:
        params3d['phi_min'] = beam_data['phiinj']*0.5

    if 'phi_max' not in params3d:
        params3d['phi_max'] = beam_data['phiinj']*1.5

    if 'label' not in params3d:
        params3d['label'] = 'AUG wall'

    if 'label' not in line_opts:
        line_opts['label'] = '$\\beta = %.2f {}^o$ & $\\beta = %.2f{}^o$'%\
            (beam_data['beta']*180/np.pi, beam_data['theta']*180/np.pi)

    line_handl= ax.plot(beam_data['beam_cart'][0, :],
                        beam_data['beam_cart'][1, :],
                        beam_data['beam_cart'][2, :],
                        '-', **line_opts)


    if diverg['plotDiv']:
        sbeam = beam_data['s_beam']
        theta_beam = np.linspace(0, 2*np.pi, num=32)
        sbeam, theta_beam = np.meshgrid(sbeam, theta_beam)
        Rprim = sbeam * np.tan(diverg['alpha']) + \
                diverg['R0'] * np.cos(diverg['alpha'])

        x = sbeam * diverg['u_inj'][0] + \
            Rprim*np.cos(theta_beam)*diverg['u_1'][0] + \
            Rprim*np.sin(theta_beam)*diverg['u_2'][0] + beam_data['origin'][0]

        y = sbeam * diverg['u_inj'][1] + \
            Rprim*np.cos(theta_beam)*diverg['u_1'][1] + \
            Rprim*np.sin(theta_beam)*diverg['u_2'][1] + beam_data['origin'][1]

        z = sbeam * diverg['u_inj'][2] + \
            Rprim*np.cos(theta_beam)*diverg['u_1'][2] + \
            Rprim*np.sin(theta_beam)*diverg['u_2'][2] + beam_data['origin'][2]
        color = line_handl[-1]._color

        ax.plot_wireframe(x, y, z, color=color, alpha=0.1)

    if drawvessel:
        ax=Lib.plt.plot_vessel(projection='3D', units='m',
                               ax=ax, params3d=params3d)

    if ax_was_none:
        ax_options['xlabel'] = 'X [m]'
        ax_options['ylabel'] = 'Y [m]'
        ax_options['zlabel'] = 'Z [m]'

        ax = Lib.plt.axis_beauty(ax, ax_options)

    return ax


class gaussian_beam:
    def __init__(self, origin: float, beta: float=0.0, theta: float=0.0,
                 meanE: float=70.0, stdE: float=0.0,
                 mass: float=libhipsim.xsection.alkMasses['Rb85'],
                 divergency: float=0.0, pinhole_size: float=0.0):

        """
        Generic class to handle beam geometries and energy injection modelling.
        The beam is modelled via a Gaussian distribution in the energy space.
        Beam divergency is also modelled via a Gaussian, centred at the
        injection direction.
        Angles must be provided in º and energies in keV.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param origin: cartesian origin (X, Y, Z) from which the beam is
        started.
        @param beta: toroidal tilting angle defined from the injection point.
        @param theta: poloidal tilting angle, defined from the injection point.
        @param meanE: average beam energy. Beam is modelled as a Gaussian.
        @param stdE: standard deviation of the energy in the Gaussian model.
        @param mass: mass of the beam species. By default, the Rb85.
        @param divergency: beam divergency.
        @param pinhole_size: this allows to set a given size initial size of
        the beam. This is the initial radius of the beam.
        """

        # Storing the input data into the class.
        self.origin = np.atleast_1d(origin)
        self.beta   = beta
        self.theta  = theta
        self.meanE  = meanE
        self.stdE   = stdE
        self.mass   = mass
        self.pinsize = pinhole_size

        self.mass_si = mass * Lib.par.amu2kg

        # Computing the beam line.
        self._beam_data = generateBeamTrajectory(start=origin, beta=beta,
                                                 theta=theta)
        self.div    = divergency

        # Setting up some flags.
        self.infsmall = False
        if self.div == 0.0:
            self.infsmall = True

        z_shift = np.array((0.0, 0.0, self.pinsize))/2.0
        self._beam_pol_up = generateBeamTrajectory(start=origin+z_shift,
                                                   beta=beta,
                                                   theta=theta+self.div)
        self._beam_pol_down = generateBeamTrajectory(start=origin-z_shift,
                                                     beta=beta,
                                                     theta=theta-self.div)

        u_perp = np.array((+self._beam_data['u_inj'][1],
                           -self._beam_data['u_inj'][2], 0.0))

        u_perp = u_perp/np.linalg.norm(u_perp)

        tor_shift = u_perp*self.pinsize
        self._beam_tor1 = generateBeamTrajectory(start=origin+tor_shift,
                                                 beta=beta+self.div,
                                                 theta=theta)
        self._beam_tor2 = generateBeamTrajectory(start=origin-tor_shift,
                                                 beta=beta-self.div,
                                                 theta=theta)

        # For plotting the beam in 3D.
        self._u1_perp = u_perp
        self._u2_perp = np.cross(self._u1_perp, self._beam_data['u_inj'])

        self.energyDispersion = False
        if self.stdE == 0.0:
            self.energyDispersion = True


    def update(self, origin: float=None, beta: float=None, theta: float=None,
               meanE: float=None, stdE: float=None, mass: float=None,
               divergency: float=None, pinhole_size: float=None):
        """
        Updates the beam data according to the input.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param origin: cartesian origin (X, Y, Z) from which the beam is
        started.
        @param beta: toroidal tilting angle defined from the injection point.
        @param theta: poloidal tilting angle, defined from the injection point.
        @param meanE: average beam energy. Beam is modelled as a Gaussian.
        @param stdE: standard deviation of the energy in the Gaussian model.
        @param mass: mass of the beam species. By default, the Rb85.
        @param divergency: beam divergency.
        @param pinhole_size: this allows to set a given size initial size of
        the beam. This is the initial radius of the beam.
        """

        recompute_flag = False

        if origin is not None:
            self.origin = np.atleast_1d(origin)
            recompute_flag = True

        if beta is not None:
            self.beta   = beta
            recompute_flag = True
        if theta is not None:
            self.theta   = theta
            recompute_flag = True

        if meanE is not None:
            self.meanE  = meanE

        if stdE is not None:
            self.stdE   = stdE

        if mass is not None:
            self.mass   = mass
            self.mass_si = mass * Lib.par.amu2kg

        if pinhole_size is not None:
            self.pinsize = pinhole_size
            recompute_flag = True

        if divergency is not None:
            self.div    = divergency
            recompute_flag = True

        # Computing the beam line.
        if recompute_flag:
            self._beam_data = generateBeamTrajectory(start=self.origin,
                                                     beta=self.beta,
                                                     theta=self.theta)
            # Setting up some flags.
            self.infsmall = False
            z_shift = np.array((0.0, 0.0, self.pinsize))
            self._beam_pol_up = generateBeamTrajectory(start=self.origin+z_shift,
                                                       beta=self.beta,
                                                       theta=self.theta+self.div)
            self._beam_pol_down = generateBeamTrajectory(start=self.origin-z_shift,
                                                         beta=self.beta,
                                                         theta=self.theta-self.div)

            u_perp = np.array((+self._beam_data['u_inj'][1],
                               -self._beam_data['u_inj'][2], 0.0))

            u_perp = u_perp/np.linalg.norm(u_perp)

            tor_shift = u_perp*self.pinsize
            self._beam_tor1 = generateBeamTrajectory(start=self.origin+tor_shift,
                                                     beta=self.beta+self.div,
                                                     theta=self.theta)
            self._beam_tor2 = generateBeamTrajectory(start=self.origin-tor_shift,
                                                     beta=self.beta-self.div,
                                                     theta=self.theta)

            # For plotting the beam in 3D.
            self._u1_perp = u_perp
            self._u2_perp = np.cross(self._u1_perp,
                                     self._beam_data['u_inj'])

        self.energyDispersion = False
        if self.stdE == 0.0:
            self.energyDispersion = True

    def plot(self, view: str='pol', ax=None, fig=None,
             **kwargs):
        """
        Plots to the given axis (if None, new ones are created).

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: view to plot the beam. pol, tor or 3d are accepted.
        @param ax: axis to plot the data. If None, new ones are created.
        @param fig: figure handler. If None, current is retrieved.
        @param kwargs: arguments to pass down to the corresponding plotting
        function.
        """

        if view.lower() == 'pol':
            divDict = {'pol_up': self._beam_pol_up,
                       'pol_down': self._beam_pol_down,
                       'plotDiv': True}
            ax = plotBeam_poloidal(beam_data=self._beam_data, ax=ax, fig=fig,
                                   **divDict, **kwargs)
        elif view.lower() == 'tor':
            divDict = { 'tor_up': self._beam_tor1,
                        'tor_down': self._beam_tor2,
                        'plotDiv': True
                      }
            ax = plotBeam_toroidal(beam_data=self._beam_data, ax=ax, fig=fig,
                                   **divDict, **kwargs)
        elif view.lower() == '3d':
            divDict = { 'u_inj': self._beam_data['u_inj'],
                        'u_1': self._u1_perp,
                        'u_2': self._u2_perp,
                        'alpha': self.div/2.0,
                        'R0': self.pinsize,
                        'plotDiv': True
                      }

            ax = plotBeam_3D(beam_data=self._beam_data, ax=ax, fig=fig,
                             diverg=divDict, **kwargs)
        else:
            raise ValueError('View %s not available!'%view)

        return ax


    def change(self, xin: float, inptype: str='cyl', outtype: str='beam_sRt'):
        """
        Wrapper allowing to change the coordinates using the beam coordinates.
        The accepted input/output coordinates are' cyl', 'cart', 'beam'

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param xin: coordinates in (inptype) to be transformed.
        @param inptype: string with the type of input coordinates.
        @param outtype: string with the type of output coordinates.
        """


        xin = np.atleast_2d(xin)
        axis_3 = np.where([(xin.shape[ii] == 3) \
                           for ii in range(len(xin.shape))])
        if len(axis_3) == 0:
            raise Exception('None of the axis is lenght 3!')

        xin = np.moveaxis(xin, axis_3, 0)
        xout = np.zeros(xin.shape)

        if inptype == outtype:
            return xin

        if inptype == 'cyl' and outtype == 'cart':
            xout[0, ...] = xin[0, ...] * np.cos(xin[1, ...])
            xout[1, ...] = xin[0, ...] * np.cos(xin[1, ...])
            xout[2, ...] = xin[2, ...]

        elif inptype == 'cart' and outtype == 'cyl':
            xout[0, ...] = np.sqrt(xin[0, ...]**2.0 + xin[1, ...]**2.0)
            xout[1, ...] = np.arctan2(xin[1, ...], xin[0, ...])
            xout[2, ...] = xin[2, ...]

        elif inptype == 'cyl' and outtype == 'beam_sRt':
            xoutaux = np.zeros(xout.shape)
            xoutaux[0, ...] = xin[0, ...] * np.cos(xin[1, ...]) - \
                              self._beam_data['origin'][0]
            xoutaux[1, ...] = xin[0, ...] * np.cos(xin[1, ...]) - \
                              self._beam_data['origin'][1]
            xoutaux[2, ...] = xin[2, ...] - \
                              self._beam_data['origin'][2]

            xoutaux_norm = np.sqrt(xoutaux[0, ...]**2 + xoutaux[2, ...]**2 + \
                                   xoutaux[1, ...]**2)
            # The coordinate along the beam line is define as the projection
            # of the vector onto the line.
            xout[0, ...] = xoutaux[0, ...] * self._beam_data['u_inj'][0] + \
                           xoutaux[1, ...] * self._beam_data['u_inj'][1] + \
                           xoutaux[2, ...] * self._beam_data['u_inj'][2]

            # Rprime is the distance to the beam axis.
            xout[1, ...] = np.sqrt(xoutaux_norm**2.0 - xout[0, ...]**2.0)

            # And the cyclic angle around the injection axis.
            xout[2, ...] = xoutaux[0, ...] * self._u1_perp + \
                           xoutaux[1, ...] * self._u1_perp + \
                           xoutaux[2, ...] * self._u1_perp
            xout[2, ...] /= xout[1, ...]

            flags = xout[1, ...] == 0.0
            xout[2, flags] = 0.0
            xout[2, ...] = np.arccos(xout[2, ...])

        elif inptype == 'cyl' and outtype == 'beam_sbT':
            xoutaux = np.zeros(xout.shape)
            xoutaux[0, ...] = xin[0, ...] * np.cos(xin[1, ...]) - \
                              self._beam_data['origin'][0]
            xoutaux[1, ...] = xin[0, ...] * np.cos(xin[1, ...]) - \
                              self._beam_data['origin'][1]
            xoutaux[2, ...] = xin[2, ...] - \
                              self._beam_data['origin'][2]

            xoutaux_norm = np.sqrt(xoutaux[0, ...]**2 + xoutaux[2, ...]**2 + \
                                   xoutaux[1, ...]**2)
            # The coordinate along the beam line is define as the projection
            # of the vector onto the line.
            xout[0, ...] = xoutaux[0, ...] * self._beam_data['u_inj'][0] + \
                           xoutaux[1, ...] * self._beam_data['u_inj'][1] + \
                           xoutaux[2, ...] * self._beam_data['u_inj'][2]

            # Rprime is the distance to the beam axis.
            rprime = np.sqrt(xoutaux_norm**2.0 - xout[0, ...]**2.0)

            xout[1, ...] = np.arccos(xoutaux_norm/xout[0, ...])

            # And the cyclic angle around the injection axis.
            xout[2, ...] = xoutaux[0, ...] * self._u1_perp + \
                           xoutaux[1, ...] * self._u1_perp + \
                           xoutaux[2, ...] * self._u1_perp
            xout[2, ...] /= rprime

            flags = rprime == 0.0
            xout[2, flags] = 0.0
            xout[2, ...] = np.arccos(xout[2, ...])

        elif inptype == 'cart' and outtype == 'cyl':
            xout[0, ...] = np.sqrt(xin[0, ...]**2.0 + xin[1, ...]**2.0)
            xout[1, ...] = np.arctan2(xin[1, ...], xin[0, ...])
            xout[2, ...] = xin[2, ...]

        elif inptype == 'beam_sRt' and outtype == 'cart':
            x = xin[0, ...] * self._beam_data['u_inj'][0] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[0] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[0] + \
                self._beam_data['origin'][0]

            y = xin[0, ...] * self._beam_data['u_inj'][1] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[1] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[1] + \
                self._beam_data['origin'][1]

            z = xin[0, ...] * self._beam_data['u_inj'][2] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[2] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[2] + \
                self._beam_data['origin'][2]

            xout[0, ...] = x
            xout[1, ...] = y
            xout[2, ...] = z

        elif inptype == 'beam_sRt' and outtype == 'cyl':
            x = xin[0, ...] * self._beam_data['u_inj'][0] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[0] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[0] + \
                self._beam_data['origin'][0]

            y = xin[0, ...] * self._beam_data['u_inj'][1] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[1] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[1] + \
                self._beam_data['origin'][1]

            z = xin[0, ...] * self._beam_data['u_inj'][2] + \
                xin[1, ...]*np.cos(xin[2, ...])* self._u1_perp[2] + \
                xin[1, ...]*np.sin(xin[2, ...])* self._u2_perp[2] + \
                self._beam_data['origin'][2]


            xout[0, ...] = np.sqrt(x**2.0 + y**2.0)
            xout[1, ...] = np.arctan2(y, x)
            xout[2, ...] = z

        elif inptype == 'beam_sbT' and outtype == 'cart':
            rprime = xin[0, ...] * np.sin(xin[1, ...])

            x = xin[0, ...] * self._beam_data['u_inj'][0] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[0] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[0] + \
                self._beam_data['origin'][0]

            y = xin[0, ...] * self._beam_data['u_inj'][1] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[1] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[1] + \
                self._beam_data['origin'][1]

            z = xin[0, ...] * self._beam_data['u_inj'][2] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[2] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[2] + \
                self._beam_data['origin'][2]

            xout[0, ...] = x
            xout[1, ...] = y
            xout[2, ...] = z

        elif inptype == 'beam_sbT' and outtype == 'cyl':
            rprime = xin[0, ...] * np.sin(xin[1, ...])

            x = xin[0, ...] * self._beam_data['u_inj'][0] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[0] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[0] + \
                self._beam_data['origin'][0]

            y = xin[0, ...] * self._beam_data['u_inj'][1] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[1] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[1] + \
                self._beam_data['origin'][1]

            z = xin[0, ...] * self._beam_data['u_inj'][2] + \
                rprime*np.cos(xin[2, ...])* self._u1_perp[2] + \
                rprime*np.sin(xin[2, ...])* self._u2_perp[2] + \
                self._beam_data['origin'][2]

            xout[0, ...] = np.sqrt(x**2.0 + y**2.0)
            xout[1, ...] = np.arctan2(y, x)
            xout[2, ...] = z

        else:
            raise Exception('Not a valid change of coordinates!')

        # Changing back the axis to its original shape.
        xin = np.moveaxis(xin, 0, axis_3)
        xout = np.moveaxis(xout, 0, axis_3)

        return xout


    def change_vel(self, vin: float, inptype: str='cyl',
                   outtype: str='beam', **kwargs):
        """
        Transformation of coordinates for the velocity using the beam
        coordinates.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param vin: velocities to transform.
        @param inptype: string with the type of input coordinates.
        @param outtype: string with the type of output coordinates.
        """

        vin = np.atleast_2d(vin)
        axis_3 = np.where([(vin.shape[ii] == 3) \
                           for ii in range(len(vin.shape))])
        if len(axis_3) == 0:
            raise Exception('None of the axis is lenght 3!')

        vin = np.moveaxis(vin, axis_3, 0)
        vout = np.zeros(vin.shape)

        if inptype == outtype:
            return vin

        if inptype == 'cyl' and outtype == 'cart':
            if 'phi' not in kwargs:
                raise Exception('Missing toroidal angle for transformation')

            vout[0, ...] = vin[0, ...]*np.cos(kwargs['phi']) - \
                           vin[1, ...]*np.sin(kwargs['phi'])

            vout[1, ...] = vin[0, ...]*np.sin(kwargs['phi']) + \
                           vin[1, ...]*np.cos(kwargs['phi'])

            vout[2, ...] = vin[2, ...]
        elif inptype == 'cart' and outtype == 'cyl':
            if 'phi' not in kwargs:
                raise Exception('Missing toroidal angle for transformation')
            vout[0, ...] = vin[0, ...]*np.cos(kwargs['phi']) + \
                           vin[1, ...]*np.sin(kwargs['phi'])

            vout[1, ...] = - vin[0, ...]*np.sin(kwargs['phi']) + \
                             vin[1, ...]*np.cos(kwargs['phi'])

            vout[2, ...] = vin[2, ...]

        elif inptype == 'beam':
            vs = vin[0, ...]*np.cos(vin[1, ...])
            v1 = vin[0, ...]*np.sin(vin[1, ...])*np.cos(vin[2, ...])
            v2 = vin[0, ...]*np.sin(vin[1, ...])*np.sin(vin[2, ...])

            u_s = self._beam_data['u_inj']
            u1  = self._u1_perp
            u2  = self._u2_perp

            vx = vs * u_s[0] + v1*u1[0] + v2*u2[0]
            vy = vs * u_s[1] + v1*u1[1] + v2*u2[1]
            vz = vs * u_s[2] + v1*u1[2] + v2*u2[2]

            if outtype == 'cart':
                vout[0, ...] = vx
                vout[1, ...] = vy
                vout[2, ...] = vz
            elif outtype == 'cyl':
                if 'phi' not in kwargs:
                    raise Exception('Missing toroidal angle for transformation')
                vout[0, ...] = vx*np.cos(kwargs['phi']) + \
                               vy*np.sin(kwargs['phi'])
                vout[1, ...] = - vx*np.sin(kwargs['phi']) + \
                                 vy*np.cos(kwargs['phi'])
                vout[2, ...] = vz

        elif inptype == 'cart' and outtype == 'beam':
            u_s = self._beam_data['u_inj']
            u1  = self._u1_perp
            u2  = self._u2_perp

            vs = vin[0, ...]*u_s[0] + vin[1, ...]*u_s[1] + vin[2, ...]*u_s[2]
            v1 = vin[0, ...]*u1[0]  + vin[1, ...]*u1[1]  + vin[2, ...]*u1[2]
            v2 = vin[0, ...]*u2[0]  + vin[1, ...]*u2[1]  + vin[2, ...]*u2[2]

            theta = np.arctan2(v2, v1)
            vel   = np.sqrt(vin[0, ...]**2.0 + vin[1, ...]**2.0 +\
                            vin[2, ...]**2.0)
            beta = np.arccos(vs/vel)

            vout[0, ...] = vel
            vout[1, ...] = beta
            vout[2, ...] = theta

        elif inptype == 'cyl' and outtype == 'beam':
            if 'phi' not in kwargs:
                    raise Exception('Missing toroidal angle for transformation')

            vx = vin[0, ...]*np.cos(kwargs['phi']) + \
                 vin[1, ...]*np.sin(kwargs['phi'])

            vy = - vin[0, ...]*np.sin(kwargs['phi']) + \
                   vin[1, ...]*np.cos(kwargs['phi'])

            vz = vin[2, ...]

            u_s = self._beam_data['u_inj']
            u1  = self._u1_perp
            u2  = self._u2_perp

            vs = vx*u_s[0] + vy*u_s[1] + vz*u_s[2]
            v1 = vx*u1[0]  + vy*u1[1]  + vz*u1[2]
            v2 = vx*u2[0]  + vy*u2[1]  + vz*u2[2]

            theta = np.arctan2(v2, v1)
            vel   = np.sqrt(vx**2.0 + vy**2.0 + vz**2.0)
            beta = np.arccos(vs/vel)

            vout[0, ...] = vel
            vout[1, ...] = beta
            vout[2, ...] = theta

        # Changing back the axis to its original shape.
        vin = np.moveaxis(vin, 0, axis_3)
        vout = np.moveaxis(vout, 0, axis_3)

        return vout


    def makebeam(self, model: int=BEAM_INF_SMALL, Nbeam = 100, Ndisk=1,
                 Rmin: float=None, Rmax: float=None):
        """
        Creates markers to sample the beam.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        if model == BEAM_INF_SMALL or model == BEAM_ENERGY_DISPERSION:
            Ndisk = 1

        if Rmin is None:
            Rmin = self._beam_data['Rbeam'].min()
        if Rmax is None:
            Rmax = self._beam_data['Rbeam'].max()


        # --- Getting the maximum beam injection position and minimum
        slimits = R2beam(u = self._beam_data['u_inj'],
                         origin = self._beam_data['origin'],
                         R = np.array((Rmin, Rmax)))

        smin = slimits.min()
        smax = slimits.max()

        if model == BEAM_INF_SMALL:
            s_data = np.linspace(smin, smax, Nbeam)
        else:
            s_data = np.atleast_1d([random.random() \
                                        for ii in range(Ndisk*Nbeam)])
            s_data = smin + (smax-smin)*s_data

        # --- Generating random points in the injection port.
        if Ndisk != 1:
            Rtilde = np.random.uniform(size=(Ndisk*Nbeam,))
            theta  = np.random.uniform(size=(Ndisk*Nbeam,))
            theta_vel = np.random.uniform(size=(Ndisk*Nbeam,))
            beta  = np.random.normal(loc=0, scale=self.div*np.pi/180.0,
                                     size=(Ndisk*Nbeam,))
            theta  = theta  * 2.0*np.pi
            Rtilde = Rtilde * self.pinsize + s_data*np.tan(beta)
        else:
            Rtilde = np.zeros(s_data.shape)
            theta  = np.zeros(s_data.shape)
            beta   = np.zeros(s_data.shape)
            theta_vel = np.zeros(s_data.shape)

        xin = np.zeros((3, len(s_data)))
        xin[0, :] = s_data
        xin[1, :] = Rtilde
        xin[2, :] = theta

        xout = self.change(xin, 'beam_sRt', 'cart')
        xout_cyl = self.change(xout, 'cart', 'cyl')

        # Generating the beam energies
        if (model == BEAM_ENERGY_DISPERSION or model == BEAM_FULLWIDTH) and \
            self.stdE != 0.0:
            Ebeam = np.random.normal(loc=self.meanE, scale=self.stdE,
                                     size=(Ndisk*Nbeam,))

            vbeam = np.sqrt(2.*Ebeam*Lib.par.ec*1e3/self.mass_si)
        else:
            vbeam = np.sqrt(2.*self.meanE*Lib.par.ec*1e3/self.mass_si)
        # Transforming the velocities into the cartesian coordinates.
        vin = np.zeros((3, len(s_data)))
        vin[0, :] = vbeam
        vin[1, :] = beta
        vin[2, :] = theta_vel

        vout = self.change_vel(vin, 'beam', 'cart')
        vout_cyl = self.change_vel(vout, 'cart', 'cyl', phi=xout_cyl[1, :])

        # We finally need the time coordinate:
        time = s_data/(vbeam*np.cos(beta))

        return xout_cyl, vout_cyl, time

# ----------------------------------------------------------------------------
# --- Scintillator and head geometry functions and classes.
# ----------------------------------------------------------------------------
class geom:
    """
    Class containing all the relevant geometric data information for the
    i-HIBP diagnostic.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    """

    def __init__(self, model3d: str=None, scint: Union[str, float]=None,
                 shot_params: dict={}, **beam_args):
        """
        Constructor of the geometry class. This will set up all the variables
        and internal functions to easily plot and manipulate the geometry
        of the system.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        @param model3d: path to the 3d file of the head model.
        @param scint: 2d model of the scintillator where the strikes will
        be recorded. Can be either the path to the file or a 2d structure.
        @param shot_params: dictionary containing shot data to retrieve
        magnetic equilibrium from a database.
        @param beam_args: beam keyword arguments to initialize the
        corresponding beam class.
        """

        # Storing the data from the 3D scintillator.
        if model3d is not None:
            if isinstance(model3d, str):
                model3d = (model3d,)
            self.head = common_geom.Geometry(files=model3d, code='ihibpsim')

        # Storing the data of the 2D scintillator.
        if scint is not None:
            if isinstance(scint, str):
                data = np.loadtxt(scint, comments='!')
            else:
                if scint.shape[1] != 3:
                    raise ValueError('The scintillator must have'+\
                                     ' 3 Cartesian coordinates!')
                data = scint

            # Separaring the points.
            self.x0 = data[0, :]
            self.x1 = data[1, :]
            self.x2 = data[2, :]

            # Computing the associated with the scintillator.
            self.v0 = self.x1 - self.x0
            self.v1 = self.x2 - self.x0

            self.x3 = self.x0 + self.v0 + self.v1

        # Loading the beam parameters.
        std_beam = { 'beam_type': 'gaussian',
                     'origin': libparms.iHIBP['port_center'],
                     'beta': libparms.iHIBP['beta_std'],
                     'theta': libparms.iHIBP['theta_std']
                   }

        std_beam.update(beam_args)

        if std_beam['beam_type'] == 'gaussian':
            std_beam.pop('beam_type')
            self.beam = gaussian_beam(**std_beam)
        else:
            raise NotImplementedError('The beam type %s not implemented'%\
                                      std_beam['beam_type'])

        # Loading the shot data, if provided.
        std_shot_data = { 'shotnumber': None,
                          'diag': None,
                          'exp': None,
                          'edition': None
                        }

        std_shot_data.update(shot_params)
        self.shotnumber = None
        self.shotdiag   = None
        self.shotexp    = None
        self.shotedit   = None
        self.Rsep = None
        self.zsep = None
        self.t_eq = None

        self.set_shot(**std_shot_data)

    def set_shot(self, shotnumber: int=None, diag: str=None, exp: str=None,
                 edition: int=None):
        """
        Set internally a reference shot to plot the separatrix and compute
        stuff of the beam.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param shotnumber: shot number to plot separatrix.
        @param diag: diagnostic name where the equilibria is stored.
        @param exp: experiment name under which the equilibirum shotfile is.
        @param edition: edition of shotfile.
        """

        if (shotnumber is None) or (diag is None) or (exp is None):
            return

        if edition is None:
            edition = 0

        self.shotnumber = shotnumber
        self.shotdiag   = diag
        self.shotexp    = exp
        self.shotedit   = edition

        # Loading the separatrix lines for all the shot times.
        tmp = Lib.dat.get_rho2rz(shot=self.shotnumber, flxlabel=[1.0],
                                 diag=self.shotdiag, exp=self.shotexp)

        self.Rsep = tmp[0]
        self.zsep = tmp[1]
        self.t_eq = tmp[2]

        return

    def __plot_sep(self, view: str='pol', ax=None, fig=None,
                   timepoint: float=None, parms: dict={}):
        """"
        Internal routine to plot the separatrix on the geometry plot. Poloidal,
        toroidal and 3D (axisymmetric) version.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: way of representing the geometry: '3d' or 'pol', 'tor' are
        available to plot.
        @param ax: axis to plot the figure.
        @param fig: figure associated to the axis where the data is being
        plot.
        @param timepoint: in case separatrix is loaded, corresponding timepoint
        can be loaded.
        @param parms: parameters for the plotting. If the 2D options are chose,
        then the options are related to the Line Object. Instead, the options
        would correspond to 3D surface plot.
        """

        if self.Rsep is None:
            raise ValueError('The shot number is not set')

        # Getting the plotting time point.
        if timepoint is None:
            tidx = [0]
        else:
            tidx = np.searchsorted(self.t_eq, timepoint)
            if not hasattr(tidx, '__iter__'):
                tidx = [tidx]

        options = { 'color': 'r',
                    'linewidth': 1.0
                  }
        options.update(parms)

        im = list()
        if view == 'pol':
            if ax is None:
                fig, ax = plt.subplots()

            for ii in tidx:
                im.append(ax.plot(self.Rsep[ii][0],
                                  self.zsep[ii][0],
                                  **options))

        elif view == 'tor':
            if ax is None:
                fig, ax = plt.subplots()

            for ii in tidx:
                Rmax = self.Rsep[ii][0].max()
                Rmin = self.Rsep[ii][0].min()
                phi = np.linspace(0, 2*np.pi)

                x = Rmax * np.cos(phi)
                y = Rmax * np.sin(phi)
                im.append(ax.plot(x, y, **options))

                x = Rmin * np.cos(phi)
                y = Rmin * np.sin(phi)
                im.append(ax.plot(x, y, **options))

        elif view == '3d':
            raise NotImplemented('Not 3D view for separatrix yet implemented.')

        return im, ax


    def __plot_scintillator(self, view: str='3d', ax=None, fig=None,
                            **kwargs):
        """
        Internal routine that plots the 2d scintillator as a surface.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: way of representing the geometry: '3d' or 'pol', 'tor' are
        available to plot.
        @param ax: axis to plot the figure.
        @param fig: figure associated to the axis where the data is being
        plot.
        @param kwargs: keyword arguments for the plotting subroutines. If a 2D
        plot view is chosen, then these would correspond to the Line
        characteristics. Otherwise, it will be properties of the 3D surface to
        plot.
        """
        if view == '3d':
            if ax is None:
                fig = plt.figure()
                fig.add_subplots(projection='3d')

            # Use Jose's function...
            raise NotImplementedError('Waiting for Jose push')
        elif view == 'pol':
            options = { 'color': 'g',
                        'alpha': 0.5
                      }

            options.update(kwargs)


            if ax is None:
                fig, ax = plt.subplots()

            # Coordinate change from cartesian to cylindrical coordinates
            # for the elements of the scintillator.
            xc0 = [np.sqrt(self.x0[0]**2 + self.x0[1]**2),
                   self.x0[2]]
            xc1 = [np.sqrt(self.x1[0]**2 + self.x1[1]**2),
                   self.x1[2]]
            xc2 = [np.sqrt(self.x2[0]**2 + self.x2[1]**2),
                   self.x2[2]]
            xc3 = [np.sqrt(self.x3[0]**2 + self.x3[1]**2),
                   self.x3[2]]

            R = [xc2[0], xc3[0], xc1[0], xc0[0]]
            z = [xc2[1], xc3[1], xc1[1], xc0[1]]

            ax.fill(R, z, **options)
        else:
            options = { 'color': 'g',
                        'alpha': 0.5
                      }

            options.update(kwargs)

            if ax is None:
                fig, ax = plt.subplots()

            x = [self.x2[0], self.x3[0], self.x1[0], self.x0[0]]
            y = [self.x2[1], self.x3[1], self.x1[1], self.x0[1]]

            ax.fill(x, y, 'g', alpha=0.5)

        return ax

    def __plot_head(self, view: str='3d', ax=None, fig=None, **kwargs):
        """
        Internal routine that plots the 3d mode of the diagnostic head.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: way of representing the geometry: only '3D' can be used.
        @param ax: axis to plot the figure.
        @param fig: figure associated to the axis where the data is being
        plot.
        @param kwargs: keyword arguments for the plotting subroutines. If a 2D
        plot view is chosen, then these would correspond to the Line
        characteristics. Otherwise, it will be properties of the 3D surface to
        plot.
        """
        if view == '3d':
            ax = self.head.plot3Dfilled(ax=ax, units='m',
                                        surface_params=kwargs,
                                        plot_pinhole=False)
        else:
            warnings.warn('Head is not plot in non-3D plots.')

        return ax


    def plot(self, view: str='3d', elements: str = 'all', ax=None, fig=None,
             timepoint: float=None):
        """
        Plots the iHIBP geometry, including the head, the scintillator plate
        and the beam injection geometry, or a part of these.

        Pablo Oyola - pablo.oyola@ipp.mpg.de

        @param view: way of representing the geometry: '3d' or 'pol', 'tor' are
        available to plot.
        @param elements: elements to plot, a combination of 'head', 'scint',
        'beam', 'separatrix', 'all' as a list.
        @param ax: axis to plot the figure.
        @param fig: figure associated to the axis where the data is being
        plot.
        @param timepoint: in case separatrix is loaded, corresponding timepoint
        can be loaded.
        """

        elements_plottable = ('head', 'scint', 'beam',
                              'separatrix', 'scintillator')

        # Checking the inputs.
        if view not in ('3d', 'pol', 'tor'):
            raise ValueError('The view can only be 3d, pol or tor.')

        if isinstance(elements, str):
            elements = (elements,)

        toplot = list()
        for iele in elements:
            if iele == 'all':
                toplot = deepcopy(elements_plottable)
                break
            elif iele not in elements_plottable:
                raise ValueError('Element %s not recognized'%iele)
            else:
                toplot.append(iele)

        # Plotting each elements.
        if 'beam' in toplot:
            ax = self.beam.plot(view=view, ax=ax, fig=fig)
        if ('scint' in toplot) or ('scintillator' in toplot):
            ax = self.__plot_scintillator(view=view, ax=ax, fig=fig)

        if 'head' in toplot:
            ax = self.__plot_head(view=view, ax=ax, fig=fig)
        if 'separatrix' in toplot:
            im, ax = self.__plot_sep(view=view, ax=ax, fig=fig,
                                     timepoint=timepoint)

        return ax

    # -------------
    # Routines to get beam characteristics.
    # -------------
    def compute_sepcross(self):
        """
        This routine will compute the cross between the beam and the separatrix
        for all the time points stored in the class for the LFS part.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        """

        if self.Rsep is None:
            raise Exception('The separatrix is not loaded. '+\
                            'Cannot compute the cross w/ separatrix')

        # Allocating space.
        rcross = np.zeros((self.t_eq.size,))
        zcross = np.zeros((self.t_eq.size,))
        for ii in range(rcross.size):
            tmp = find_2D_intersection(self.Rsep[ii][0],
                                       self.zsep[ii][0],
                                       self.beam._beam_data['Rbeam'],
                                       self.beam._beam_data['zbeam'])
            tmp = np.array(tmp)

            if tmp.shape[1] > 1:
                Rmaxidx = tmp[0,:].argmax()
                tmp = tmp[:, Rmaxidx]

            rcross[ii] = tmp[0]
            zcross[ii] = tmp[1]

        return self.t_eq, rcross, zcross
