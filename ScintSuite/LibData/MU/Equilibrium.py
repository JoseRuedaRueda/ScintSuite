"""Routines for the magnetic equilibrium"""
import numpy as np
from pyEquilibrium.equilibrium import equilibrium
import pyuda
import logging

logger = logging.getLogger('ScintSuite.MUequilibrium')



def get_mag_field(shot: int, Rin, zin, time: float, flag_MSE = None, **kwargs):
    """
    Get MU magnetic field

    Lina Velarde: lvelarde@us.es

    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)

    Note2: MU FILD1 is located around z=0.159m

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this)
    :param  flag_MSE: If True, will use MSE-constrained eq. If False, will use
        regular reconstruction. Default is None. This means the MSE-constrained
        will be attempted but if it does not exist or the data is bad will read
        regular EFIT reconstruction.

    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]
    @TODO: Include the sign of Bpol
    """
    if isinstance(time, (list, np.ndarray)):
        pass
    else:  # it should be just a number
        time = np.array([time])
    
    if not isinstance(Rin, np.ndarray):
        Rin = np.array([Rin])
    if not isinstance(zin, np.ndarray):
        zin = np.array([zin])

    br = np.zeros((time.shape[0], Rin.shape[0]))
    bz = np.zeros((time.shape[0], Rin.shape[0]))
    bp = np.zeros((time.shape[0], Rin.shape[0]))
    bt = np.zeros((time.shape[0], Rin.shape[0]))

    if flag_MSE is not None:
        if flag_MSE == True:
            logger.info('MSE-constrained eq will be used')
            client = pyuda.Client()
            t = client.get('/epq/time', shot).data
            r = client.get('/epq/output/profiles2d/r', shot).data
            z = client.get('/epq/output/profiles2d/z', shot).data
            for ii in range(len(time)):
                idxt = (np.abs(t-time[ii])).argmin()
                for jj in range(len(Rin)):
                    idxr = (np.abs(r-Rin[jj])).argmin()
                    idxz = (np.abs(z-zin[jj])).argmin()
                    br[ii, jj] = client.get('/epq/output/profiles2d/br', shot).data[idxt,idxr,idxz]
                    bz[ii, jj] = client.get('/epq/output/profiles2d/bz', shot).data[idxt,idxr,idxz]
                    bp[ii, jj] = client.get('/epq/output/profiles2d/bpol', shot).data[idxt,idxr,idxz]
                    bt[ii, jj] = client.get('/epq/output/profiles2d/bphi', shot).data[idxt,idxr,idxz]
        elif flag_MSE == False:
            logger.warning('Regular eq, not MSE-constrained, will be used')
            for ii in range(len(time)):
                efit_eq = equilibrium(
                    shot="/common/uda-scratch/lkogan/efitpp_eshed/epm{:0>6}.nc".
                    format(shot) if shot < 44000 else shot,
                    device='MASTU', time=time[ii]
                )
                for jj in range(len(Rin)):
                    br[ii, jj] = efit_eq.BR(Rin[jj], zin[jj])
                    bz[ii, jj] = efit_eq.BZ(Rin[jj], zin[jj])
                    bp[ii, jj] = efit_eq.Bp(Rin[jj], zin[jj])
                    bt[ii, jj] = efit_eq.Bt(Rin[jj], zin[jj])
    else:
        try:
            logger.info('No equilibrium source specified. Will attempt MSE-constrained')
            client = pyuda.Client()
            t = client.get('/epq/time', shot).data
            r = client.get('/epq/output/profiles2d/r', shot).data
            z = client.get('/epq/output/profiles2d/z', shot).data
            for ii in range(len(time)):
                idxt = (np.abs(t-time[ii])).argmin()
                for jj in range(len(Rin)):
                    idxr = (np.abs(r-Rin[jj])).argmin()
                    idxz = (np.abs(z-zin[jj])).argmin()
                    br[ii, jj] = client.get('/epq/output/profiles2d/br', shot).data[idxt,idxr,idxz]
                    bz[ii, jj] = client.get('/epq/output/profiles2d/bz', shot).data[idxt,idxr,idxz]
                    bp[ii, jj] = client.get('/epq/output/profiles2d/bpol', shot).data[idxt,idxr,idxz]
                    bt[ii, jj] = client.get('/epq/output/profiles2d/bphi', shot).data[idxt,idxr,idxz]
        except pyuda.ServerException:
            logger.warning('MSE-constrained data BAD or non existent. Will use regular equilibrium')
            for ii in range(len(time)):
                efit_eq = equilibrium(
                    shot="/common/uda-scratch/lkogan/efitpp_eshed/epm{:0>6}.nc".
                    format(shot) if shot < 44000 else shot,
                    device='MASTU', time=time[ii]
                )
                for jj in range(len(Rin)):
                    br[ii, jj] = efit_eq.BR(Rin[jj], zin[jj])
                    bz[ii, jj] = efit_eq.BZ(Rin[jj], zin[jj])
                    bp[ii, jj] = efit_eq.Bp(Rin[jj], zin[jj])
                    bt[ii, jj] = efit_eq.Bt(Rin[jj], zin[jj])

    return br, bz, bt, bp
