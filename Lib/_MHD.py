"""
Class to calculate the mode frecuencies

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import Lib.LibData as ssdat
import scipy.constants as cnt
import logging

# --- Initilise the objects
logger = logging.getLogger('ScintSuite.MHD')


# -----------------------------------------------------------------------------
# --- Main class
# -----------------------------------------------------------------------------
class MHDmode():
    def __init__(self, shot: int = 41091, mi=2.013*cnt.m_u):
        """
        Initialise the class and read the plasma profiles
        """
        logger.warning('XX: Assuming ne=ni')
        # --- Densities
        self._ne = ssdat.get_ne(shotnumber=shot, xArrayOutput=True)
        self._ni = self._ne.copy()

        # --- Temperatures:
        self._te = ssdat.get_Te(shotnumber=shot, xArrayOutput=True)
        try:
            self._ti = ssdat.get_Ti(shotnumber=shot, diag='IDI',
                                    xArrayOutput=True)
        except:
            logger.warning('XX: Not func Ti, using Te=Ti')
            self._ti = self._te.copy()
        # --- q-profile:
        self._q = ssdat.get_q_profile(shot, xArrayOutput=True)
        # --- Other data:
        self._basic = ssdat.get_shot_basics(shot)
        self._R0 = xr.Dataset()
        self._R0['data'] = xr.DataArray(
            self._basic['Rmag'], dims='t',
            coords={'t': self._basic['time']})
        self._kappa = xr.Dataset()
        self._kappa['data'] = xr.DataArray(
            self._basic['k'], dims='t',
            coords={'t': self._basic['time']})
        self._B0 = xr.Dataset()
        self._B0['data'] = xr.DataArray(
            self._basic['bt0'], dims='t',
            coords={'t': self._basic['bttime']})
        # --- Now interpolate everything in the time/rho basis of ne
        self._ni = self._ni.interp(t=self._ne['t'], rho=self._ne['rho'])
        self._te = self._te.interp(t=self._ne['t'], rho=self._ne['rho'])
        self._ti = self._ti.interp(t=self._ne['t'], rho=self._ne['rho'])
        self._q = self._q.interp(t=self._ne['t'], rho=self._ne['rho'])
        self._R0 = self._R0.interp(t=self._ne['t'])
        self._B0 = self._B0.interp(t=self._ne['t'])
        self._kappa = self._kappa.interp(t=self._ne['t'])

        # --- Precalculate some stuff:
        self._va0 = self._B0 / np.sqrt(cnt.mu_0 * mi * self._ni)

        # --- Alocate space for the frequencies
        self.freq = xr.Dataset()

        # --- Save the mass info
        self._mi = mi

        # --- Calculate the frequencies
        self._calcGAMfreq()
        self._calcTAEfreq()

    def _calcGAMfreq(self):
        """
        Evaluate the GAM frequency

        following expresion (1) of W.W. Heidbrink Nucl. Fusion 61 (2021)
        """
        self.freq['GAM'] = np.sqrt(
            1.0 / 2.0 / cnt.pi**2 / self._mi / self._R0['data']**2
            * (self._te['data'] + 7.0/4.0 * self._ti['data'])
            * (1.0 + 1.0/2.0/self._q['data']**2)
            * 2.0 / (self._kappa['data']**2 + 1)*cnt.eV)
        self.freq['GAM'].attrs['long_name'] = '$f_{GAM}$'
        self.freq['GAM'].attrs['units'] = 'Hz'

    def _calcTAEfreq(self):
        """
        Evaluate the central frequency of the TAE gap

        following eq (1) of W.W. Heidbrink PRL 71 1993
        """
        self.freq['TAE'] = \
            self._va0['data']/4.0/cnt.pi/self._q['data']/self._R0['data']

    def plot(self, var: str = 'GAM', rho=0.0, ax=None, line_params={},
             units: str = 'kHz'):
        """
        Plot the mode frequency

        Jore Rueda: jrrueda@us.es

        @param var: Mode to be plotted
        @param rho: list with the desired values of rho where to plot
        @param ax: axes where to plot
        @param line_params: Line parameters for matplotlib.pyplot.plot
        @param units: units for the frequency: accpted: Hz, kHz, rad

        @TODO: Include doppler shift correction
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
        line_opitons.update(line_params)
        # --- Check the rho
        try:
            len(rho)
        except TypeError:
            rho = np.array([rho])
        # --- Prepare the axis
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_ylabel('$f_{%s}$ [%s]' % (var, units))
            ax.set_xlabel('Time [s]')
        # --- Plot the line
        for r in rho:
            if r < self.freq[var]['rho'][0] or r < self.freq[var]['rho'][1]:
                text = 'Requested point outise the rho interval, skipping
                logger.warning('XX: %s', text)
            data = self.freq[var].sel(rho=r, method='nearest')
            ax.plot(data['t'], data.values*factor[units], **line_opitons)
        plt.draw()
        plt.show()
