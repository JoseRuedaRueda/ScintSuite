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

BEAM_INF_SMALL = 0
BEAM_ANGULAR_DIVERGENCY = 1
BEAM_ENERGY_DISPERSION = 2
BEAM_FULLWIDTH = 3

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
                           Rmin: float=1.65, Rmax: float=None, Ns: int = 128):
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
                      ax_options: dict={}, units: str='m'):
    """
    Projects onto the poloidal plane the beam line.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot.
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
        ax.fill_between(pol_up['Rbeam'], pol_up['zbeam'], pol_down['zbeam'],
                        linewidth=0.0, color=color, alpha=0.1)
        
    
    if drawvessel:
        ax=Lib.plt.plot_vessel(projection='pol', units=units,
                               color='k', linewidth=0.75, ax=ax)
    
    if ax_was_none:
        ax_options['ratio'] = 'equal'
        ax_options['xlabel'] = 'Major radius R [m]'
        ax_options['ylabel'] = 'Height z [m]'
        ax_options['grid'] = 'both'
        
        ax = Lib.plt.axis_beauty(ax, ax_options)
        
    return ax


def plotBeam_toroidal(beam_data: dict, ax=None, fig=None, 
                      tor_up: dict={}, tor_down: dict={}, plotDiv: bool=False,
                      drawvessel: bool=True, line_opts: dict={},
                      ax_options: dict={}, units: str='m'):
    """
    Projects onto the toroidal plane the beam line.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot.
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
        x = (tor_up['beam_cart'][0, 0], tor_up['beam_cart'][0, -1], 
             tor_down['beam_cart'][0, -1], tor_down['beam_cart'][0, 0])
        y = (tor_up['beam_cart'][1, 0], tor_up['beam_cart'][1, -1], 
             tor_down['beam_cart'][1, -1], tor_down['beam_cart'][1, 0])
        ax.fill(x, y, color=color, alpha=0.1, linewidth=0.0)
    
    if drawvessel:
        ax=Lib.plt.plot_vessel(projection='tor', units=units,
                               color='k', linewidth=0.75, ax=ax)
    
    if ax_was_none:
        ax_options['ratio'] = 'equal'
        ax_options['xlabel'] = 'X [m]'
        ax_options['ylabel'] = 'Y [m]'
        ax_options['grid'] = 'both'
        
        ax = Lib.plt.axis_beauty(ax, ax_options)
        
    return ax


def plotBeam_3D(beam_data: dict, ax=None, fig=None, 
                drawvessel: bool=True, diverg: dict= {}, line_opts: dict={},
                ax_options: dict={}, params3d: dict={}, units: str='m'):
    """
    Plots the 3D vessel and the beam line in 3D.
    
    Pablo Oyola - pablo.oyola@ipp.mpg.de
    
    @param beam_data: beam data as obtained from the generateBeamTrajectory.
    @param ax: axis to plot. If None, new ones are created.
    @param fig: figure handler. If None, gcf is used to retrieve them.
    @param drawvessel: Plots the vessel into the axis. True by default.
    @param line_opts: options to send down to plt.plot.
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
        ax=Lib.plt.plot_vessel(projection='3D', units=units,
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
                 mass: float=Lib.iHIBP.LibHIBP_crossSections.alkMasses['Rb85'],
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
        else:
            z_shift = np.array((0.0, 0.0, self.pinsize))
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
            if not self.infsmall:
                divDict = { 'pol_up': self._beam_pol_up,
                            'pol_down': self._beam_pol_down,
                            'plotDiv': True
                          }
            else:
                divDict = {}
            ax = plotBeam_poloidal(beam_data=self._beam_data, ax=ax, fig=fig,
                                   **divDict, **kwargs)
        elif view.lower() == 'tor':
            if not self.infsmall:
                divDict = { 'tor_up': self._beam_tor1,
                            'tor_down': self._beam_tor2,
                            'plotDiv': True
                          }
            else:
                divDict = {}
            ax = plotBeam_toroidal(beam_data=self._beam_data, ax=ax, fig=fig,
                                   **divDict, **kwargs)
        elif view.lower() == '3d':
            if not self.infsmall:
                divDict = { 'u_inj': self._beam_data['u_inj'],
                            'u_1': self._u1_perp,
                            'u_2': self._u2_perp,
                            'alpha': self.div/2.0,
                            'R0': self.pinsize,
                            'plotDiv': True
                          }
            else:
                divDict = {}
            
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
        
        @param inp: string with the type of input coordinates.
        @param out: string with the type of output coordinates.
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
        Transformation of coordinates for the velocity.
        
        Pablo Oyola - pablo.oyola@ipp.mpg.de
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
        
        s_data = np.atleast_1d([random.random() \
                                    for ii in np.arange(Ndisk*Nbeam)])
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
        
        # --- Computing the velocity-space volume:
        idx_sorting_beta = np.argsort(beta)
        bbeam_sorted = beta[idx_sorting_beta]
        print(bbeam_sorted.shape)
        
        dbbeam_sorted = np.zeros(beta.shape)
        db = np.zeros(beta.shape)
        
        dbbeam = bbeam_sorted[2:] - bbeam_sorted[:-2]
        dbbeam_idx0 = bbeam_sorted[1] - bbeam_sorted[0]
        dbbeam_idx_end = bbeam_sorted[-1] - bbeam_sorted[-2]
        dbbeam_sorted[0] = dbbeam_idx0
        dbbeam_sorted[-1] = dbbeam_idx_end
        dbbeam_sorted[1:-1] = dbbeam
        db[idx_sorting_beta] = dbbeam_sorted
        
        return beta, db