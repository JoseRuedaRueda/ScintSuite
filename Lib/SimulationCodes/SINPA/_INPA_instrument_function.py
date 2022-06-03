"""
Contain the class to prepare and apply the INPA instrument function

HIGHLY under development. Do not Use if you do not know what are you doing
"""
import numpy as np
import matplotlib.pyplot as plt
from Lib.SimulationCodes.FIDASIM._FBM_class import FBM
from Lib._SideFunctions import createGrid1D
from scipy.sparse import lil_matrix
import warnings
warnings.filterwarnings("error", append=True)


class INPA_IF():
    """
    INPA instrument function

    yes, contract Instrument Function as IF is maybe not the best in a code
    but... I like it

    Jose Rueda: jrrueda@us.es

    Public methods:
        -

    Introduced in version 0.10.0
    """

    def __init__(self, strikes, grid_scintillator_options: dict = None,
                 grid_FBM_options: dict = None,  fbm_file: str = None,
                 variables: str = 'EpRz_ER'):
        """
        Inialize the class.

        Jose Rueda: jrrueda@us.es

        @param strikes object. Notice that the remapping should have been done!
        @param grid_options: grid options with the minimum for each variable
            and grid spacing. Optional, not needed if fbm_file is not none, as
            the grid will be loaded from there
        @param fbm_file: path to the fbm_file from FIDASIM4, to load the grid
        @param variables: variabled for the IF calculation. xxxx_yy, xxxx is
            the set of 4 variables in the fast-ion space and yy the one in the
            INPA scintillator one. As of today, only EpRz_ER is implemented as
            an option.
        """
        # --- Load the strike points
        self._strike_points = strikes

        # --- Load/construct the grid
        if grid_FBM_options is None:
            fbm = FBM(fbm_file)
            rmax = fbm['nr'] * fbm['dr'] + fbm['rmin']
            zmax = fbm['nz'] * fbm['dz'] + fbm['zmin']
            grid_FBM_options = {
                'e0min': fbm['emin'],
                'e0max': fbm['emax'],
                'de0': fbm['dE'],
                'pitchmin': fbm['pmin'],
                'pitchmax': fbm['pmax'],
                'dpitch': fbm['dp'],
                'z0min': fbm['zmin'],
                'z0max': zmax,
                'dz0': fbm['dz'],
                'R0min': fbm['rmin'],
                'R0max': rmax,
                'dR0': fbm['dr'],
            }
        self._grid_FBM_options = grid_FBM_options
        # Construct the grid
        if variables.lower() == 'eprz_er':
            var = ['e0', 'pitch0', 'R0', 'z0']
            var_scint = ['remap_e0', 'remap_R0']
            self._strike_var_names = ['e0', 'pitch0', 'R0', 'z0',
                                      'remap_e0', 'remap_R0']
        self._grid_FBM = dict.fromkeys(var)
        self._grid_scintillator = dict.fromkeys(var_scint)
        s2 = 1
        for i in var:
            n, edges = createGrid1D(
                grid_FBM_options[i + 'min'],
                grid_FBM_options[i + 'max'],
                grid_FBM_options['d' + i])
            center = 0.5 * (edges[1:] + edges[:-1])
            self._grid_FBM[i] = {
                'n': n,
                'edges': edges.copy(),
                'centers': center.copy()
            }
            s2 *= n
        s1 = 1
        for i in var_scint:
            n, edges = createGrid1D(
                grid_scintillator_options[i + 'min'],
                grid_scintillator_options[i + 'max'],
                grid_scintillator_options['d' + i])
            center = 0.5 * (edges[1:] + edges[:-1])
            self._grid_scintillator[i] = {
                'n': n,
                'edges': edges.copy(),
                'centers': center.copy()
            }
            s1 *= n
        # save the future size of the sparese matrix
        self.shape = (s1, s2)

    def _calculate_6D_IF(self):
        """
        Calculate the 6D instrument_function
        """
        # First get the indexes
        index = []
        for key in self._strike_var_names:
            index.append(self._strike_points.header['info'][key]['i'])
        nalpha = self._grid_scintillator[self._strike_var_names[4]]['n']
        nbeta = self._grid_scintillator[self._strike_var_names[5]]['n']

        # Proceed with the calculation:
        ikind = self._strike_points.header['info']['kind']['i']
        iw = self._strike_points.header['info']['weight']['i']
        self._matrix = {5: None, 6: None}

        for kind in (5, 6):  # Lopp over passive and active signal
            # Get the markers corresponging to the kind
            flags_kind = \
                np.round(self._strike_points.data[0, 0][:, ikind]) == kind
            if flags_kind.sum() == 0:
                continue
            # Proceed to calculate
            sparse = lil_matrix(self.shape)
            edges = [self._grid_FBM[k]['edges'] for k in self._strike_var_names[0:4]]
            for i1 in range(self.shape[0]):
                # Get the index of the scintillator space cell
                ibeta = np.mod(i1, nbeta)
                ialpha = int((i1 - ibeta)/nbeta)
                # Get the limits of that cell
                alpha_min = self._grid_scintillator[self._strike_var_names[4]]['edges'][ialpha]
                alpha_max = self._grid_scintillator[self._strike_var_names[4]]['edges'][ialpha+1]
                beta_min = self._grid_scintillator[self._strike_var_names[5]]['edges'][ibeta]
                beta_max = self._grid_scintillator[self._strike_var_names[5]]['edges'][ibeta+1]
                # Get which markers correspond to this cell
                flags_alpha = \
                    (alpha_min <= self._strike_points.data[0, 0][:, index[4]])\
                    * (self._strike_points.data[0, 0][:, index[4]] < alpha_max)
                flags_beta = \
                    (beta_min <= self._strike_points.data[0, 0][:, index[5]])\
                    * (self._strike_points.data[0, 0][:, index[5]] < beta_max)
                # Get the total flags
                total_flags = flags_kind * flags_alpha * flags_beta
                # Perform the 4D histogram
                dummy = self._strike_points.data[0, 0][total_flags, ...][:, index[0:4]]
                if total_flags.sum() == 0:
                    continue
                try:
                    raise Exception('oye que cada casilla esta normalizada a 1')
                    H, edges_hist = np.histogramdd(
                        dummy,
                        bins=edges,
                        weights=self._strike_points.data[0, 0][total_flags, iw],
                        density=True
                    )
                    # Save the data in the sparse
                    sparse[i1, :] = H.flatten()
                except RuntimeWarning:
                    print(ibeta, ialpha)
                    print('alpha:', alpha_min, alpha_max)
                    print('beta:', beta_min, beta_max)
                    print('fa:', flags_alpha.sum())
                    print('fb:', flags_beta.sum())
                    print('fab:', (flags_alpha * flags_beta).sum())
                    print('ft:', total_flags.sum())
                    print(self._strike_points.data[0, 0][total_flags, ...][:, index])
                    print(self._strike_points.data[0, 0][total_flags, :])
                    print(H.shape)
                    print(edges[0].shape)
                    print(edges[1].shape)
                    print(edges[2].shape)
                    print(edges[3].shape)
                    Q = H.sum(axis=(1,2,3))
                    e = 0.5*(edges[0][:-1] + edges[0][1:])
                    plt.plot(e, Q)
                    print(Q[Q>0])
                if i1 < 124:
                    print(ibeta, ialpha)
                    print('alpha:', alpha_min, alpha_max)
                    print('beta:', beta_min, beta_max)
                    print('fa:', flags_alpha.sum())
                    print('fb:', flags_beta.sum())
                    print('fab:', (flags_alpha * flags_beta).sum())
                    print('ft:', total_flags.sum())
                    print(self._strike_points.data[0, 0][total_flags, ...][:, index])
                    print(self._strike_points.data[0, 0][total_flags, :])
                    print(H.shape)
                    print(edges[0].shape)
                    print(edges[1].shape)
                    print(edges[2].shape)
                    print(edges[3].shape)
                    Q = H.sum(axis=(1,2,3))
                    e = 0.5*(edges[0][:-1] + edges[0][1:])
                    plt.plot(e, Q)
                    print(Q[Q>0])
            # Save the data
            self._matrix[kind] = sparse.copy()
