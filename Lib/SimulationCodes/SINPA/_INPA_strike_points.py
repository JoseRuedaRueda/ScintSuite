"""
Class containing INPA strike points
"""

from Lib.SimulationCodes.Common.strikes import Strikes
import Lib.LibData as ssdat
import Lib.errors as errors
import numpy as np


class INPAStrikes(Strikes):
    """
    Strike object tunned for INPA strike points

    Public methods not present in the parent class:

        calculatePitch: calculate the pitch of the markers
    """

    def caclualtePitch(self, Boptions: dict = {}, IPBtSign: float = -1.0):
        """
        Calculate the pitch of the markers

        Jose Rueda: jrrueda@us.es

        Note: This function only works for INPA markers
        Note2: This is mainly though to be used for FIDASIM markers, no mapping
        for these guys, there is only one set of data, so is nice. If you use
        it for the mapping markers, there would be ngyr x nR set of data and
        the magnetic field shotfile will be open those number of times. Not the
        end of the world, but just to have it in mind. A small work arround
        could be added if needed, althoug that would imply something like:
        if machine == 'AUG'... which I would like to avoid

        @param Boptions: extra parameters for the calculation of the magnetic
            field
        """
        # Only works for INPA markers, return error if we try with FILD
        if self.header['FILDSIMmode']:
            raise errors.NotValidInput('Hey! This is only for INPA')
        # See if we need to overwrite
        if 'pitch0' in self.header['info'].keys():
            print('The pitch values are there, we will overwrite them')
            overwrite = True
            ipitch0 = self.header['info']['pitch0']['i']
        else:
            overwrite = False
        # Get the index for the needed variables
        ir = self.header['info']['R0']['i']          # get the index
        iz = self.header['info']['z0']['i']
        ix = self.header['info']['x0']['i']
        iy = self.header['info']['y0']['i']
        ivz = self.header['info']['vz0']['i']
        ivx = self.header['info']['vx0']['i']
        ivy = self.header['info']['vy0']['i']
        for ig in range(self.header['ngyr']):
            for ia in range(self.header['nXI']):
                if self.header['counters'][ia, ig] > 0:
                    phi = np.arctan2(self.data[ia, ig][:, iy],
                                     self.data[ia, ig][:, ix])
                    br, bz, bt, bp =\
                        ssdat.get_mag_field(self.header['shot'],
                                            self.data[ia, ig][:, ir],
                                            self.data[ia, ig][:, iz],
                                            time=self.header['time'],
                                            **Boptions)
                    bx = br*np.cos(phi) - bt*np.sin(phi)
                    by = - br*np.cos(phi) + bt*np.cos(phi)
                    b = np.sqrt(bx**2 + by**2 + bz**2).squeeze()
                    v = np.sqrt(self.data[ia, ig][:, ivx]**2
                                + self.data[ia, ig][:, ivy]**2
                                + self.data[ia, ig][:, ivz]**2)
                    pitch = (self.data[ia, ig][:, ivx] * bx
                             + self.data[ia, ig][:, ivy] * by
                             + self.data[ia, ig][:, ivz] * bz).squeeze()
                    pitch /= v*b*IPBtSign
                    if overwrite:
                        self.data[ia, ig][:, ipitch0] = pitch.copy()
                    else:
                        self.data[ia, ig] = \
                            np.append(self.data[ia, ig],
                                      np.atleast_2d(pitch.copy()).T, axis=1)

        if not overwrite:
            Old_number_colums = len(self.header['info'])
            extra_column = {
                'pitch0': {
                    'i': Old_number_colums,  # Column index in the file
                    'units': ' []',  # Units
                    'longName': 'Pitch',
                    'shortName': '$\\lambda_{0}$',
                },
            }
            self.header['info'].update(extra_column)
