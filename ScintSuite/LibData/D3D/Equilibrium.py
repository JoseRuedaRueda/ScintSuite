"""
Module to interact with the D3D magnetic equilibrium

Jose Rueda Rueda

"""
import numpy as np
import xarray as xr
from ScintSuite.LibData.MDSplus.efit import EFIT

def get_mag_field(shot: int, Rin, zin, time: float, efit_tree: str = 'efit02',
                  MDSserver = 'atlas.gat.com', 
                  precalculateInterpolators = True,
                  **kwargs):
    """
    Get the magnetic field from the D3D EFIT tree
    
    :param shot: Shot number
    :param Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param time: Array of times where we want to calculate the field [in s]
    :param efit_tree: Name of the EFIT tree to use (default: 'efit02')
    :param kwargs: Extra arguments to pass to the EFIT class
    
    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]
    @TODO: Include the sign of Bpol
    """
    # Open the EFIT tree
    efit = EFIT(shot=shot, efit=efit_tree, 
                server=MDSserver,
                precalculateInterpolators=precalculateInterpolators,
                **kwargs)
    br, bz, bt = efit.Bfield(r=Rin, z=zin, time=time*1000.0)
    bp = np.sqrt(br**2 + bz**2)
    return br, bz, bt, bp

def get_separatrix(shot: int, time: float, efit_tree: str = 'efit02',
                  MDSserver = 'atlas.gat.com',):
    """
    Get the separatrix from the D3D EFIT tree
    
    :param shot: Shot number
    :param time: Array of times where we want to calculate the separatrix
        The closest time to this will be used [in seconds]
    :param efit_tree: Name of the EFIT tree to use (default: 'efit02')
    :param MDSserver: Server where to get the data
    
    :return Rsep: R position of the separatrix
    :return zsep: z position of the separatrix
    """
    # Open the EFIT tree
    efit = EFIT(shot=shot, efit=efit_tree, server=MDSserver,
                precalculateInterpolators=False)
    # Get the separatrix
    boundary = efit['BDRY']
    # Get the time
    itime = efit.closest_time(time*1000.0)
    # Get the separatrix
    Rsep = boundary[itime, :, 0]
    zsep = boundary[itime, :, 1]
    return Rsep, zsep
    
def get_magnetic_axis(shot: int, time: float = None, efit_tree: str = 'efit02',
                      MDSserver = 'atlas.gat.com',):
    """
    Get the separatrix from the D3D EFIT tree
    
    :param shot: Shot number
    :param time: Array of times where we want to calculate the separatrix
        The closest time to this will be used [in seconds]
    :param efit_tree: Name of the EFIT tree to use (default: 'efit02')
    :param MDSserver: Server where to get the data
    
    :return Rsep: R position of the separatrix
    :return zsep: z position of the separatrix
    """
    # Open the EFIT tree
    efit = EFIT(shot=shot, efit=efit_tree, server=MDSserver,
                precalculateInterpolators=False)
    # Get the time
    if time is not None:
        itime = efit.closest_time(time*1000.0)
        # Get the axis
        Rsep = efit['RMAXIS'][itime]
        zsep = efit['ZMAXIS'][itime]
    else:
        Rsep = efit['RMAXIS']
        zsep = efit['ZMAXIS']
    return Rsep, zsep

def get_q_profile(shot: int, rho=np.linspace(0, 1), 
                  time: float = None, efit_tree: str = 'efit02',
                  MDSserver = 'atlas.gat.com',
                  **kwargs):
    """
    Get the q profile from the D3D EFIT tree
    
    Jose Rueda Rueda:
    :param shot: Shot number
    :param time: Time [s] where we want the profiles, if None, all available times are returned
    """
    efit = EFIT(shot=shot, efit=efit_tree, server=MDSserver,
            precalculateInterpolators=False)
    if time is None: #ms for the efit object, s for the output
        time = efit['GTIME']
        time2 = time/1000.0
    else:
        time = time*1000.0
        time2 = time
    TT, RR = np.meshgrid(time, rho, indexing='ij')
    q = xr.Dataset()
    q['data'] = xr.DataArray(efit.q(time=TT.flatten(), rhopol=RR.flatten()).reshape(TT.shape).T,
                             dims=['rho', 't'],
                             coords={'t': time2, 'rho': rho})
    q['data'].attrs['long_name'] = 'Safety factor'
    q['rho'].attrs['long_name'] = r'$\rho_p$'
    q['t'].attrs['units'] = 's'
    q['t'].attrs['long_name'] = 'Time'
    q.attrs['shot'] = shot
    q.attrs['diag'] = efit_tree
    
    return q

def get_shot_basics(shot, efit_tree: str = 'efit02',
                  MDSserver = 'atlas.gat.com',
                  **kwargs):
    """
    Get the basic information for a shot
    
    This is not as complete as the AUG case, only has the minimum for the MHD
    Object to work
            self._R0 = xr.Dataset()
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
    """
    efit = EFIT(shot=shot, efit=efit_tree, server=MDSserver,
                precalculateInterpolators=False)
    data = {
        'Rmag':efit['RMAXIS'],
        'time':efit['GTIME']/1000.0,
        'ahor':efit['AMINOR'],
        'k':efit['KAPPA'],
        'bt0':efit['BCENTR'],
        'bttime':efit['GTIME']/1000.0
    }
    return data
    