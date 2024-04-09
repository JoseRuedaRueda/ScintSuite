"""
Class to calculate the analytical mode frecuencies.

Jose Rueda Rueda: jrrueda@us.es
"""
import logging
import numpy as np
import xarray as xr
import ScintSuite.errors as errors
import matplotlib.pyplot as plt
import ScintSuite.LibData as ssdat 
import scipy.constants as cnt
from typing import Optional
from tqdm import tqdm
from ScintSuite._Machine import machine 
# --- Initilise the objects
logger = logging.getLogger('ScintSuite.MHD')

# -----------------------------------------------------------------------------
# %% SAW continum
# -----------------------------------------------------------------------------
class SAWc():
    """
    Shear Alfven Wave continum object 
    """
    def __init__(self, q, R0, ahor, va0,
                 ntor: int, t: Optional[tuple]= None, mpol = np.arange(257),
                 rotation: Optional[xr.DataArray] = None, 
                 correct_by_plasma_rotation: bool = True):
        # Save the data
        self._q = q
        self._R0 = R0
        self._ahor = ahor
        self._va0 = va0
        self._rotation = rotation
        #
        self.ntor = ntor
        self.t = t
        self.mpol = mpol
        self.corrections = {
            'plasma_rotation': correct_by_plasma_rotation
        }
        # Calculate the things
        self._calcSAW()

    def _calcSAW(self):
        """
        Obtain SAW continuum in its simpler form from the analytical equation.

        The expression is obtained from:
        Fu, G. Y., & Van Dam, J. W. (1989).
        Physics of Fluids B, 1(10), 1949?1952.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        Adapted to the object by Jose Rueda-Rueda: jrrueda@us.es

        @param rpsi: radial magnetic coordinate.
        @param qpsi: q-profile evaluated in 'rpsi'.
        @param ntor: list of toroidal mode numbers.
        @param mpol: list of poloidal mode numbers.
        @param aminor: value of the minor radius.
        @param Rmajor: value of the major radius.
        """
        if self.t is None:
            Qpsi = np.abs(self._q.data)
            RMajor = self._R0['data']
            Aminor = self._ahor['data']
            VA = self._va0['data']
        else:
            Qpsi = np.abs(self._q.data.sel(t=slice(self.t[0], self.t[1])))
            RMajor = self._R0['data'].sel(t=slice(self.t[0], self.t[1]))
            Aminor = self._ahor['data'].sel(t=slice(self.t[0], self.t[1]))
            VA = self._va0['data'].sel(t=slice(self.t[0], self.t[1]))
        rpsi = self._q.data.rho.values
        for jt, t in enumerate(tqdm(VA.t)):
            # --- Get inputs
            qpsi = Qpsi.isel(t=jt).values
            Rmajor = RMajor.isel(t=jt).values
            aminor = Aminor.isel(t=jt).values
            vA = VA.isel(t=jt).values

            inv_asp_rat = 3.0/2.0 * aminor/Rmajor
            invAspRat2 = inv_asp_rat**2.0

            # --- Transforming the inputs into vectors:
            ntor = np.atleast_1d(self.ntor)
            mpol = np.atleast_1d(self.mpol)

            # --- Generating output:
            gqq, gmm = np.meshgrid(qpsi, mpol)
            grr, gmm = np.meshgrid(rpsi, mpol)

            # omega_max = np.zeros([len(rpsi), len(mpol)-1, len(ntor)])
            # omega_min = np.zeros([len(rpsi), len(ntor)])
            if jt == 0:
                f_max = np.zeros([len(rpsi), len(mpol)-1, len(ntor), Qpsi.t.size])
                f_min = np.zeros([len(rpsi), len(ntor), Qpsi.t.size])

            for jn, intor in enumerate(ntor):
                kpara = (gmm - intor*gqq)/(Rmajor*gqq)
                kpara2 = kpara**2.0
                diffe2 = (kpara2[1:, :] - kpara2[:-1, :])**2.0 + \
                        4.0 * kpara2[:-1, :] * kpara2[1:, :] * invAspRat2 *\
                            rpsi**2.0

                diffe2 = np.sqrt(diffe2)

                omega1 = kpara2[:-1, :] + kpara2[1:, :] + diffe2
                omega2 = kpara2[:-1, :] + kpara2[1:, :] - diffe2

                omega1 /= 2.0*(1.0 - invAspRat2*rpsi**2.0)
                omega2 /= 2.0*(1.0 - invAspRat2*rpsi**2.0)

                omega1 = np.sqrt(omega1)
                omega2 = np.sqrt(omega2)
                omega2 = np.nanmin(omega2, axis=0)

                f_max[:, :, jn, jt] = (omega1 * vA/(2.0*np.pi)).T
                f_min[:, jn, jt] = omega2 * vA/(2.0*np.pi)
        output = xr.Dataset()
        output['fMax'] = xr.DataArray(f_max.squeeze(), dims=('rho', 'm', 't'),
                                      coords={'rho': self._q.rho, 
                                              'm': mpol[:-1],
                                              't': Qpsi.t})
        output['fMax'].attrs['desc'] = 'Max gap frequency'
        output['fMax'].attrs['long_name'] = 'Freq.'
        output['fMax'].attrs['units'] = 'Hz'

        output['fMin'] = xr.DataArray(f_min.squeeze(), dims=('rho','t'))
        output['fMin'].attrs['desc'] = 'Min gap frequency'
        output['fMin'].attrs['long_name'] = 'Freq.'
        output['fMin'].attrs['units'] = 'Hz'

        output['n'] = ntor

        if 'plasma_rotation' in self.corrections:
            if self.corrections['plasma_rotation']:
                if self.t is None:
                    correction = self._rotation.data.copy()
                else: 
                    correction = \
                        self._rotation.data.sel(
                            t=slice(self.t[0], self.t[1])).copy()
                correction /= 2.0*np.pi
                correction *= self.ntor
                output['fMinRotationCorrected'] = output['fMin'] + correction
                output['fMaxRotationCorrected'] = output['fMax'] + correction
                output['fMinRotationCorrected'].attrs['desc'] ='Plasma rotation applied'
                output['fMaxRotationCorrected'].attrs['desc'] ='Plasma rotation applied'
            
        self.data = output
    
    def plot(self, t: float, ax: Optional[plt.Axes] = None, 
             line_params: dict={}, 
             units: str = 'kHz', rotationCorrected=True,)-> plt.Axes:
        """
        Plot the SAW on the given axis
        """
        # Prepare the line params
        plot_options = {
            'lw': '0.75',
            'color': 'k',
            'ls': '--',
            'label': '__no_name__'
        }
        plot_options.update(line_params)
        # Prepare the scale factor
        factors = {'Hz': 1.0, 'rad/s': 2.0*np.pi,
                   'kHz': 0.001, 'krad/s': 0.002*np.pi,
                   '1/h': 3600.0}
        # Open the axis
        if ax is None:
            fig, ax = plt.subplots()
            created = True
        else:
            fig = ax.get_figure()
            created = False
            oldXlim = ax.get_xlim()
            oldYlim = ax.get_ylim()
        # select the key
        if rotationCorrected:
            keyToPlot = 'RotationCorrected'
            corrected = True
        else:
            keyToPlot = ''
            corrected = False
        # Plot
        ax.plot(self.data.rho, 
                self.data['fMin%s'%keyToPlot].sel(t=t, method='nearest')
                *factors[units],
                **plot_options)
        ax.plot(self.data.rho, 
                self.data['fMax%s'%keyToPlot].sel(t=t, method='nearest')
                *factors[units],
                **plot_options)
        if corrected:
            color = plot_options.pop('color')
            colors = ['r', 'g', 'y', 'k']
            for c in colors:
                if c != color:
                    plot_options['color'] = c
                    break
            ax.plot(self.data.rho, 
                    self.data['fMin%s'%keyToPlot].sel(t=t, method='nearest')
                    *factors[units] - self.data['fMin'].sel(t=t, method='nearest')
                    *factors[units],
                    **plot_options)
        
        # Ax beauty
        if created:
            ax.set_ylabel('Freq. [%s]'%units)
            ax.set_xlabel(self.data.rho.long_name)
            ax.set_ylim(0, 1.75*self.data.fMin.mean()*factors[units])
        else:
            ax.set_xlim(oldXlim)
            ax.set_ylim(oldYlim)
        fig.show()
        return ax


# -----------------------------------------------------------------------------
# --- Main class
# -----------------------------------------------------------------------------
class MHDmode():
    """
    Read plasma profile and calculate analytical frequencies.

    :param shot: (int) shot number
    :param mi: (float) background ion mass, SI,
    :param loadTi: (bool) to load the ion temperature, if not, Te=Ti assumed
    :param Zimp: (float) charge of the impurities, to calculate ni

    :Example of use:

    >>> import Lib as ss
    >>> modes = ss.mhd.MHDmode(41090)
    """

    def __init__(self, shot: int = 41091, mi=2.013*cnt.m_u,
                 loadTi: bool = True, calcNi: bool = True,
                 Zimp: float = 6.0):

        
        # --- Densities
        self._ne = ssdat.get_ne(shotnumber=shot, xArrayOutput=True)
        # See if there is Zeff information
        if calcNi:
            try:
                self._zeff = ssdat.get_Zeff(shot)
                self._zeff = self._zeff.interp(t=self._ne['t'], rho=self._ne['rho'])
                self._ni = self._ne.copy()
                self._ni['data'] = self._ne['data'] * (1.0 - (self._zeff['data'] - 1.0)/(Zimp - 1.0))
                self._ni['uncertainty'] = self._ni['data'] * (
                    self._ne['uncertainty']/self._ne['data'] +
                    1.0/(Zimp - 1.0)/(1.0 - (self._zeff['data'] - 1.0)/(Zimp - 1.0))*self._zeff['uncertainty'])
            except errors.DatabaseError:
                logger.warning('Using ni=ne, no Zeff found in database')
                self._ni = self._ne.copy()
        else: 
            self._ni = self._ne.copy()

        # --- Temperatures:
        self._te = ssdat.get_Te(shotnumber=shot, xArrayOutput=True)
        if loadTi:
            try:
                self._ti = ssdat.get_Ti(shot=shot,
                                        xArrayOutput=True)
            except:
                logger.warning('XX: Not func Ti, using Te=Ti')
                self._ti = self._te.copy()
        else:
            self._ti = self._te.copy()
        # --- q-profile:
        self._q = ssdat.get_q_profile(shot, xArrayOutput=True)
        # --- Other data:
        self._basic = ssdat.get_shot_basics(shot)
        self._R0 = xr.Dataset()
        self._R0['data'] = xr.DataArray(
            self._basic['Rmag'], dims='t',
            coords={'t': self._basic['time']})
        self._ahor = xr.Dataset()
        self._ahor['data'] = xr.DataArray(
            self._basic['ahor'], dims='t',
            coords={'t': self._basic['time']})
        self._kappa = xr.Dataset()
        self._kappa['data'] = xr.DataArray(
            self._basic['k'], dims='t',
            coords={'t': self._basic['time']})
        self._B0 = xr.Dataset()
        self._B0['data'] = xr.DataArray(
            self._basic['bt0'], dims='t',
            coords={'t': self._basic['bttime']})
        # ---- Plasma rotation
        try:
            self._rotation = ssdat.get_tor_rotation(shot, xArrayOutput=True)
        except:
            self._rotation = None
            logger.warning('Not found toroidal rotation, no doppler shift')
        # --- Now interpolate everything in the time/rho basis of ne
        # self._ni = self._ni.interp(t=self._ne['t'], rho=self._ne['rho'],
        #                            method="cubic")

        #AJVV sometimes there are nans when trying to interpolate 
        if not np.isnan(self._te.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")).any().data:
            self._te = self._te.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")
        else:
            self._te = self._te.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="linear")        
        if np.isnan(self._te.data).any().data:
            self._te = self._te.fillna(0) 
        #self._te = self._te.interp(t=self._ne['t'], rho=self._ne['rho'],
        #                           method="cubic")

        #AJVV sometimes there are nans when trying to interpolate 
        if not np.isnan(self._ti.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")).any().data:
            self._ti = self._ti.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")
        else:
            self._ti = self._ti.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="linear")        
        if np.isnan(self._ti.data).any().data:
            self._ti = self._ti.fillna(0) 
            
        #self._ti = self._ti.interp(t=self._ne['t'], rho=self._ne['rho'],
        #                           method="cubic")

        #AJVV sometimes there are nans when trying to interpolate the q profile 
        #on ne timebase. So first try a linear interpolation, then check again
        #for any nans
        if not np.isnan(self._q.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")).any().data:
            self._q = self._q.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="cubic")
        else:
            self._q = self._q.interp(t=self._ne['t'], rho=self._ne['rho'],        
                                    method="linear")        
        if np.isnan(self._q.data).any().data:
            self._q = self._q.fillna(0) #self._q.dropna(dim='rho')
            
        self._R0 = self._R0.interp(t=self._ne['t'], method="cubic")
        self._ahor = self._ahor.interp(t=self._ne['t'], method="cubic")
        self._B0 = self._B0.interp(t=self._ne['t'], method="cubic")
        self._kappa = self._kappa.interp(t=self._ne['t'], method="cubic")
        #JPS: same as with the interpolation of the q-profile
        if self._rotation is not None:
            if not np.isnan(self._rotation.interp(t=self._ne['t'],
                                                   rho=self._ne['rho'],
                                                   method="cubic",
                                                   kwargs={'fill_value': 0.0})).any().data:
                self._rotation = self._rotation.interp(t=self._ne['t'],
                                                   rho=self._ne['rho'],
                                                   method="cubic",
                                                   kwargs={'fill_value': 0.0})
            else:
                self._rotation = self._rotation.interp(t=self._ne['t'],
                                                   rho=self._ne['rho'],
                                                   method="linear",
                                                   kwargs={'fill_value': 0.0})

        # --- Precalculate some stuff:
        if self._B0.data.mean() < 0.0:
            factor = -1.0
        else:
            factor = 1.0
        self._va0 = factor*self._B0 / np.sqrt(cnt.mu_0 * mi * self._ni*1.0e19)

        # --- Alocate space for the frequencies
        self.freq = xr.Dataset()

        # --- Save the mass info
        self._mi = mi

        # --- Calculate the frequencies
        self._calcGAMfreq()
        self._calcTAEfreq()
        self._calcEAEfreq()
        self._calcMTMfreq()
        self._calcKBMfreq()
        self._calcBAEfreq()
        
    def _calcMTMfreq(self) -> None:
        """
        Evaluate the MTM frequency.

        Following expresion (1) of W.W. Heidbrink Nucl. Fusion 61 (2021)
        """
        n = 2
        ne = self._ne['data']*1.0e19
        te = self._te['data']*cnt.eV
        Lne = (1/ne)*np.abs(ne.differentiate('rho')) 
        Lte = (1/te)*np.abs(te.differentiate('rho'))
        cs = np.sqrt(self._zeff['data']*te/self._mi)
        omega_i = np.abs(self._B0*cnt.e/self._mi)
        self.freq['MTM'] = 0.95*((1 / self._ahor['data']**2) * (n*self._q['data'])  
            / np.sqrt(ne['rho']) * cs**2 / omega_i['data']  
            * (Lne + Lte)) / (2*cnt.pi)
        
        self.freq['MTM'].attrs['long_name'] = '$f_{MTM}$'
        self.freq['MTM'].attrs['units'] = 'Hz'
        
    def _calcKBMfreq(self) -> None:
        """
        Evaluate the MTM frequency.

        Following expresion (1) of W.W. Heidbrink Nucl. Fusion 61 (2021)
        """
        n = 2
        ni = self._ni['data']*1.0e19
        ti = self._ti['data']*cnt.eV
        Lni = (1/ni)*np.abs(ni.differentiate('rho')) 
        Lti = (1/(ti))*np.abs(ti.differentiate('rho'))
        cs = np.sqrt(self._zeff['data']*ti/self._mi)
        omega_i = np.abs(self._zeff['data']*self._B0*cnt.e/self._mi)
        self.freq['KBM'] = 0.5*((1 / self._ahor['data']**2) * (n*self._q['data'])  
            / np.sqrt(ni['rho']) * cs**2 / omega_i['data']  
            * (Lni + Lti)) / (2*cnt.pi)
        
        
        
        self.freq['KBM'].attrs['long_name'] = '$f_{KBM}$'
        self.freq['KBM'].attrs['units'] = 'Hz'
        
    def _calcBAEfreq(self) -> None:
        """
        Evaluate the MTM frequency.

        Following expresion (1) of W.W. Heidbrink Nucl. Fusion 61 (2021)
        """
        
        te = self._te['data']*cnt.eV
        ti = self._ti['data']*cnt.eV
        
        vthermal_i = np.sqrt(ti / self._mi)
                
        self.freq['BAE'] = (vthermal_i / self._R0['data']
                    * np.sqrt(7/4 + te / ti)
                    ) / (2*cnt.pi)
        
        
        
        self.freq['BAE'].attrs['long_name'] = '$f_{KBM}$'
        self.freq['BAE'].attrs['units'] = 'Hz'
        
    def _calcGAMfreq(self) -> None:
        """
        Evaluate the GAM frequency.

        Following expresion (1) of W.W. Heidbrink Nucl. Fusion 61 (2021)
        """
        self.freq['GAM'] = np.sqrt(
            1.0 / 2.0 / cnt.pi**2 / self._mi / self._R0['data']**2
            * (self._te['data'] + 7.0/4.0 * self._ti['data'])
            * (1.0 + 1.0/2.0/self._q['data']**2)
            * 2.0 / (self._kappa['data']**2 + 1)*cnt.eV)
        self.freq['GAM'].attrs['long_name'] = '$f_{GAM}$'
        self.freq['GAM'].attrs['units'] = 'Hz'

    def _calcTAEfreq(self) -> None:
        """
        Evaluate the central frequency of the TAE gap

        Following eq (1) of W.W. Heidbrink PRL 71 1993
        """
        self.freq['TAE'] = \
            self._va0['data']/4.0/cnt.pi/self._q['data']/self._R0['data']
        if self.freq['TAE'].mean() < 0.0:
            self.freq['TAE'] *= -1.0  # the q profile was defined as negative

    def _calcEAEfreq(self) -> None:
        """
        Evaluate the central frequency of the EAE gap

        Following eq (8) of L. VILLARD NUCLEAR FUSION, Vo1.32,N0.10 (1992)
        """
        self.freq['EAE'] = \
            self._va0['data']/2.0/cnt.pi/self._q['data']/self._R0['data']

    def _calcRSAEfreq(self, n: int, m: int) -> None:
        """
        Evaluate the central frequency of the RSAE in the zero pressure limit

        Following eq from M. A. Van Zeeland, et al. Phys. Plasmas 14, 2007

        Warning, it does not check that the q profile is actually sheared, just
        take the minimum value
        """
        qmin = self._q.min(dim='rho')
        self.freq['RSAE'] = (m - n * qmin['data']) *\
            self._va0['data']/2.0/cnt.pi/qmin['data']/self._R0['data']

    def getSAWcontinuum(self, ntor: int, t: Optional[list]=None,
                        mpol = np.arange(257),
                        correct_by_plasma_rotation: bool = True)-> SAWc:
        """
        Obtain SAW continuum in its simpler form from the analytical equation.
        """

        return SAWc(self._q, self._R0, self._ahor, self._va0, ntor, t, mpol,
                    rotation=self._rotation,
                    correct_by_plasma_rotation=correct_by_plasma_rotation)

    def plot(self, var: str = 'GAM', rho: float = 0.0, ax=None, line_params={},
             units: str = 'kHz', t: tuple = None, smooth: int = 0,
             n: float=None)->plt.Axes:
        """
        Plot the mode frequency.

        Jore Rueda: jrrueda@us.es

        :param  var: Mode to be plotted
        :param  rho: list with the desired values of rho where to plot
        :param  ax: axes where to plot
        :param  line_params: Line parameters for matplotlib.pyplot.plot
        :param  units: units for the frequency: accpted: Hz, kHz, rad
        """
        factor = {
            'kHz': 0.001,
            'rad': 1.0/2.0/cnt.pi,
            'Hz': 1.0
        }
        # --- Initialise plotting settings
        line_opitons = {
            'color': 'w',
            'linestyle': '--',
            'alpha': 0.35
        }
        if ax is None:
            line_opitons['color'] = 'k'
        line_opitons.update(line_params)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_ylabel('$f_{%s}$ [%s]' % (var, units))
            ax.set_xlabel('Time [s]')
        # --- Check the rho
        if not type(rho) == xr.core.dataarray.DataArray:
            try:
                len(rho)
            except TypeError:
                rho = np.array([rho])
                logger.warning('rho not correct!')
                
            # --- Plot the line
            for r in rho:
                if r < self.freq[var]['rho'][0] or r < self.freq[var]['rho'][1]:
                    text = 'Requested point outise the rho interval, skipping'
                    logger.warning('XX: %s', text)
                data = self.freq[var].sel(rho=r, method='nearest')
                if t is None:
                    flags = np.ones(data.t.size, dtype=bool)
                else:
                    flags = (data.t > t[0]) * (data.t < t[1])
                dataToPlot = data[flags].copy()
                # Now smooth if needed
                if smooth > 0:
                    dataToPlot = dataToPlot.rolling(t=smooth, center=True).mean()
                if n is not None:
                    if self._rotation is None:
                        raise Exception('You want Doppler shift but there is no f')
                    else:
                        correction = self._rotation.data.sel(rho=r,
                                                            method='nearest').copy()
                        correction /= 2*np.pi
                        correction = correction[flags]
                        correction *= n
                else:
                    correction = np.zeros(dataToPlot.shape)

                ax.plot(dataToPlot.t, (dataToPlot + correction) * factor[units],
                        **line_opitons)
        else:
            dataToPlot = np.zeros(self.freq[var].t.size)
            correction = self._rotation.data/2.0/np.pi*n
            for jt, t in enumerate(self.freq[var].t):
                r = rho.interp(t=t).values
                if np.isnan(r):
                    continue
                if r < self.freq[var]['rho'][0] or r < self.freq[var]['rho'][1]:
                    text = 'Requested point outise the rho interval, skipping'
                    logger.warning('XX: %s', text)
                    logger.warning('Skipped!!!')
                dataToPlot[jt] = (self.freq[var].sel(rho=r, t=t, method='nearest')+\
                    correction.sel(rho=r, t=t, method='nearest')).values
                print(r,dataToPlot[jt])
            ax.plot(self.freq[var].t, dataToPlot* factor[units], **line_opitons)


        
        plt.draw()
        plt.show()
        return ax


