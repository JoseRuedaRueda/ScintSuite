"""
Class to calculate the mode frecuencies

Jose Rueda Rueda: jrrueda@us.es
"""
import numpy as np
import xarray as xr
import Lib.errors as errors
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
    """
    Read plasma profile and calculate analytical frequencies.

    :param shot: (int) shot number
    :param mi: (float) background ion mass, SI,
    :param loadTi: (bool) to load the ion temperature, if not, Te=Ti assumed

    :Example of use:

    >>> import Lib as ss
    >>> modes = ss.mhd.MHDmode(41090)
    """

    def __init__(self, shot: int = 41091, mi=2.013*cnt.m_u,
                 loadTi: bool = True):

        logger.warning('XX: Assuming ne=ni')
        # --- Densities
        self._ne = ssdat.get_ne(shotnumber=shot, xArrayOutput=True)
        self._ni = self._ne.copy()

        # --- Temperatures:
        self._te = ssdat.get_Te(shotnumber=shot, xArrayOutput=True)
        if loadTi:
            try:
                self._ti = ssdat.get_Ti(shot=shot, diag='IDI',
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
            self._rotation = ssdat.get_tor_rotation_idi(shot, xArrayOutput=True)
        except errors.DatabaseError:
            self.rotation = None
            logger.warning('Not found toroidal rotation, no doppler shift')
        # --- Now interpolate everything in the time/rho basis of ne
        self._ni = self._ni.interp(t=self._ne['t'], rho=self._ne['rho'],
                                   method="cubic")
        self._te = self._te.interp(t=self._ne['t'], rho=self._ne['rho'],
                                   method="cubic")

        self._ti = self._ti.interp(t=self._ne['t'], rho=self._ne['rho'],
                                   method="cubic")

        self._q = self._q.interp(t=self._ne['t'], rho=self._ne['rho'],
                                 method="cubic")
        self._R0 = self._R0.interp(t=self._ne['t'], method="cubic")
        self._ahor = self._ahor.interp(t=self._ne['t'], method="cubic")
        self._B0 = self._B0.interp(t=self._ne['t'], method="cubic")
        self._kappa = self._kappa.interp(t=self._ne['t'], method="cubic")
        if self._rotation is not None:
            self._rotation = self._rotation.interp(t=self._ne['t'],
                                                   method="cubic")

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

    def _calcGAMfreq(self) -> None:
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

    def _calcTAEfreq(self) -> None:
        """
        Evaluate the central frequency of the TAE gap

        following eq (1) of W.W. Heidbrink PRL 71 1993
        """
        self.freq['TAE'] = \
            self._va0['data']/4.0/cnt.pi/self._q['data']/self._R0['data']
        if self.freq['TAE'].mean() < 0.0:
            self.freq['TAE'] *= -1.0  # the q profile was defined as negative

    def _calcEAEfreq(self) -> None:
        """
        Evaluate the central frequency of the EAE gap

        following eq (8) of L. VILLARD NUCLEAR FUSION, Vo1.32,N0.10 (1992)
        """
        self.freq['EAE'] = \
            self._va0['data']/2.0/cnt.pi/self._q['data']/self._R0['data']

    def _calcRSAEfreq(self, n: int, m: int) -> None:
        """
        Evaluate the central frequency of the RSAE in the zero pressure limit

        following eq from M. A. Van Zeeland, et al. Phys. Plasmas 14, 2007

        Warning, it does not check that the q profile is actually sheared, just
        take the minimum value
        """
        qmin = self._q.min(dim='rho')
        self.freq['RSAE'] = (m - n * qmin['data']) *\
            self._va0['data']/2.0/cnt.pi/qmin['data']/self._R0['data']

    def getSAWcontinuum(self, ntor: int, t: float, mpol = np.arange(6))->dict:
        """
        Obtain SAW continuum in its simpler form from the analytical equation.

        The expression is obtained from:
        Fu, G. Y., & Van Dam, J. W. (1989).
        Physics of Fluids B, 1(10), 1949?1952.

        Pablo Oyola - pablo.oyola@ipp.mpg.de
        Adapted to the object by Jose Rueda

        @param rpsi: radial magnetic coordinate.
        @param qpsi: q-profile evaluated in 'rpsi'.
        @param ntor: list of toroidal mode numbers.
        @param mpol: list of poloidal mode numbers.
        @param aminor: value of the minor radius.
        @param Rmajor: value of the major radius.
        """
        # --- Get inputs
        qpsi = np.abs(self._q.data.sel(t=t, method='nearest')).values
        rpsi = self._q.data.rho.values
        Rmajor = self._R0['data'].sel(t=t, method='nearest').values
        aminor = self._ahor['data'].sel(t=t, method='nearest').values
        vA = self._va0['data'].sel(t=t, method='nearest').values

        inv_asp_rat = 3.0/2.0 * aminor/Rmajor
        invAspRat2 = inv_asp_rat**2.0

        # --- Transforming the inputs into vectors:
        ntor = np.atleast_1d(ntor)
        mpol = np.atleast_1d(mpol)

        # --- Generating output:
        gqq, gmm = np.meshgrid(qpsi, mpol)
        grr, gmm = np.meshgrid(rpsi, mpol)

        omega_max = np.zeros([len(rpsi), len(mpol)-1, len(ntor)])
        omega_min = np.zeros([len(rpsi), len(ntor)])
        logger.debug('Omega max shape: %i, %i, %i' % omega_max.shape)
        logger.debug('Omega min shape: %i, %i' % omega_min.shape)
        for jn, intor in enumerate(ntor):
            kpara = (gmm - intor*gqq)/(Rmajor*gqq)
            kpara2 = kpara**2.0
            diffe2 = (kpara2[1:, :] - kpara2[:-1, :])**2.0 + \
                      4.0 * kpara2[:-1, :] * kpara2[1:, :] * invAspRat2 *\
                          rpsi**2.0

            diffe2 = np.sqrt(diffe2)
            if jn == 0:
                logger.debug('kpara shape: %i, %i' % kpara.shape)
                logger.debug('diffe2 shape: %i, %i' % diffe2.shape)

            omega1 = kpara2[:-1, :] + kpara2[1:, :] + diffe2
            omega2 = kpara2[:-1, :] + kpara2[1:, :] - diffe2

            omega1 /= 2.0*(1.0 - invAspRat2*rpsi**2.0)
            omega2 /= 2.0*(1.0 - invAspRat2*rpsi**2.0)
            if jn == 0:
                logger.debug('omega1: %i, %i'%omega1.shape)
                logger.debug('omega2: %i, %i'%omega2.shape)

            omega1 = np.sqrt(omega1)
            omega2 = np.sqrt(omega2)
            omega2 = np.nanmin(omega2, axis=0)

            omega_max[:, :, jn] = (omega1 * vA/(2.0*np.pi)).T
            omega_min[:, jn] = omega2 * vA/(2.0*np.pi)

        output = {
            'f_max': omega_max.squeeze(),
            'f_min': omega_min.squeeze(),
            'ntor': ntor,
            'mpol': mpol,
            'rpsi': rpsi
        }
        return output

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
        if ax is None:
            line_opitons['color'] = 'k'
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
                dataToPlot = dataToPlot.rolling(t=smooth).mean()
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
        plt.draw()
        plt.show()
        return ax
