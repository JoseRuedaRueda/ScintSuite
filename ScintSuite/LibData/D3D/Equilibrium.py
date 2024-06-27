"""
Module to interact with the D3D magnetic equilibrium

Jose Rueda Rueda

"""
import numpy as np
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
    
def get_magnetic_axis(shot: int, time: float, efit_tree: str = 'efit02',
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
    itime = efit.closest_time(time*1000.0)
    # Get the axis
    Rsep = efit['RMAXIS'][itime]
    zsep = efit['ZMAXIS'][itime]
    return Rsep, zsep