"""
Strike map for the INPA diagnostic

Jose Rueda: jrrueda@us.es
"""
import numpy as np
import Lib.LibData as ssdat
from Lib._Machine import machine
from Lib._basicVariable import BasicVariable
from Lib._StrikeMap._FILD_INPA_ParentStrikeMap import FILDINPA_Smap


class Ismap(FILDINPA_Smap):
    """Strike map for the INPA diagnostic"""

    def __init__(self, file: str = None, variables_to_remap: tuple = None,
                 code: str = None, theta: float = None, phi: float = None,
                 GeomID: str = 'iAUG01', verbose: bool = True,
                 decimals: int = 1, rho_pol: BasicVariable = None,
                 rho_tor: BasicVariable = None):
        """
        Initialise the INPA strike map.

        This is essentially the same initialisation than the parent strike map
        but it allows to externally set the rho values. This is useful for the
        case of the remap, where we would need to recalculate the associated
        rho in each time point, which can be time consuming, to avoid this,
        we just calculate all time points before the remap (so just one
        equilibrium must be loaded) and then include the rho to each loaded map
        """
        FILDINPA_Smap.__init__(self, file=file,
                               variables_to_remap=variables_to_remap,
                               code=code, theta=theta, phi=phi, GeomID=GeomID,
                               verbose=verbose, decimals=decimals)

        if rho_pol is not None:
            self._data['rho_pol'] = rho_pol
        if rho_tor is not None:
            self._data['rho_tor'] = rho_pol

    def getRho(self, shot, time, coord: str = 'rho_pol',
               extra_options: dict = {},):
        """
        Get the rho coordinates associated to each strike point
        """
        # Initialise the equilibrium options
        if machine.lower() == 'aug':
            options = {
                'diag': 'EQH',
                'exp': 'AUGD',
                'ed': 0,
            }
        else:
            options = {}
        options.update(extra_options)

        rho = ssdat.get_rho(shot, self('R0'),
                            self('z0'), time=time,
                            coord_out=coord,
                            **options).squeeze()
        rmag, zmag, time = ssdat.get_mag_axis(shot=shot, time=time)

        flags = self('R0') < rmag
        rho[flags] *= -1.0

        self._data[coord] = BasicVariable(
            name=coord,
            units='',
            data=rho,
        )
