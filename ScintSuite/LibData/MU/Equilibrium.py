"""Routines for the magnetic equilibrium"""
import numpy as np
from pyEquilibrium.equilibrium import equilibrium
import pyuda
import logging

logger = logging.getLogger('ScintSuite.MUequilibrium')

def get_mag_field(shot: int, Rin, zin, time: float, flag_MSE = None, **kwargs):
    """
    Get MU magnetic field - loopless remake of the old function

    Theo Gheorghiu: theo.gheorghiu@ukaea.uk

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

    ## if single values are passed, ensure they're in array format
    if isinstance(time, (list, np.ndarray)):
        pass
    else:  # it should be just a number
        time = np.array([time])
    
    if not isinstance(Rin, np.ndarray):
        Rin = np.array([Rin])
    if not isinstance(zin, np.ndarray):
        zin = np.array([zin])

    ## allocate arrays to fill
    br = np.zeros((time.shape[0], Rin.shape[0]))
    bz = np.zeros((time.shape[0], Rin.shape[0]))
    bp = np.zeros((time.shape[0], Rin.shape[0]))
    bt = np.zeros((time.shape[0], Rin.shape[0]))


    ## create pyuda client
    client = pyuda.Client()

    ## check the various cases; use either mse constrained 
    ## equilibria or magnetics only
    if flag_MSE is not None:
        if flag_MSE == True:
            logger.info('MSE-constrained eq will be used')
            ## read epq data 
            t = client.get('/epq/time', shot).data
            r = client.get('/epq/output/profiles2d/r', shot).data
            z = client.get('/epq/output/profiles2d/z', shot).data
            br_all = client.get('/epq/output/profiles2d/br', shot).data
            bz_all = client.get('/epq/output/profiles2d/bz', shot).data
            bpol_all = client.get('/epq/output/profiles2d/bpol', shot).data
            bphi_all = client.get('/epq/output/profiles2d/bphi', shot).data

            ## obtain all indices needed 
            idxt = np.abs(t[None, :] - time[:, None]).argmin(axis=-1)
            idxr = np.abs(r[None, :] - Rin[:, None]).argmin(axis=-1)
            idxz = np.abs(z[None, :] - zin[:, None]).argmin(axis=-1)

            ## use double indexing to allocate fields simultaneously
            br = br_all[idxt, ...][:, idxr, idxz]
            bz = bz_all[idxt, ...][:, idxr, idxz]
            bp = bpol_all[idxt, ...][:, idxr, idxz]
            bt = bphi_all[idxt, ...][:, idxr, idxz]        

        elif flag_MSE == False:
            logger.warning('Regular eq, not MSE-constrained, will be used')
            ## access epm files..
            t = client.get('/epm/time', shot).data
            r = client.get('/epm/output/profiles2d/r', shot).data
            z = client.get('/epm/output/profiles2d/z', shot).data
            br_all = client.get('/epm/output/profiles2d/br', shot).data
            bz_all = client.get('/epm/output/profiles2d/bz', shot).data
            bpol_all = client.get('/epm/output/profiles2d/bpol', shot).data
            bphi_all = client.get('/epm/output/profiles2d/bphi', shot).data

            idxt = np.abs(t[None, :] - time[:, None]).argmin(axis=-1)
            idxr = np.abs(r[None, :] - Rin[:, None]).argmin(axis=-1)
            idxz = np.abs(z[None, :] - zin[:, None]).argmin(axis=-1)

            br = br_all[idxt, ...][:, idxr, idxz]
            bz = bz_all[idxt, ...][:, idxr, idxz]
            bp = bpol_all[idxt, ...][:, idxr, idxz]
            bt = bphi_all[idxt, ...][:, idxr, idxz]
    
    else: 
        try:
            logger.info('No equilibrium source specified. Will attempt MSE-constrained')
            ## read epq data 
            t = client.get('/epq/time', shot).data
            r = client.get('/epq/output/profiles2d/r', shot).data
            z = client.get('/epq/output/profiles2d/z', shot).data
            br_all = client.get('/epq/output/profiles2d/br', shot).data
            bz_all = client.get('/epq/output/profiles2d/bz', shot).data
            bpol_all = client.get('/epq/output/profiles2d/bpol', shot).data
            bphi_all = client.get('/epq/output/profiles2d/bphi', shot).data

            ## obtain all indices needed 
            idxt = np.abs(t[None, :] - time[:, None]).argmin(axis=-1)
            idxr = np.abs(r[None, :] - Rin[:, None]).argmin(axis=-1)
            idxz = np.abs(z[None, :] - zin[:, None]).argmin(axis=-1)

            ## use double indexing to allocate fields simultaneously
            br = br_all[idxt, ...][:, idxr, idxz]
            bz = bz_all[idxt, ...][:, idxr, idxz]
            bp = bpol_all[idxt, ...][:, idxr, idxz]
            bt = bphi_all[idxt, ...][:, idxr, idxz]  

        except pyuda.ServerException:
            logger.warning('MSE-constrained data BAD or non existent. Will use regular equilibrium')
            ## access epm files..
            t = client.get('/epm/time', shot).data
            r = client.get('/epm/output/profiles2d/r', shot).data
            z = client.get('/epm/output/profiles2d/z', shot).data
            br_all = client.get('/epm/output/profiles2d/br', shot).data
            bz_all = client.get('/epm/output/profiles2d/bz', shot).data
            bpol_all = client.get('/epm/output/profiles2d/bpol', shot).data
            bphi_all = client.get('/epm/output/profiles2d/bphi', shot).data

            idxt = np.abs(t[None, :] - time[:, None]).argmin(axis=-1)
            idxr = np.abs(r[None, :] - Rin[:, None]).argmin(axis=-1)
            idxz = np.abs(z[None, :] - zin[:, None]).argmin(axis=-1)

            br = br_all[idxt, ...][:, idxr, idxz]
            bz = bz_all[idxt, ...][:, idxr, idxz]
            bp = bpol_all[idxt, ...][:, idxr, idxz]
            bt = bphi_all[idxt, ...][:, idxr, idxz]

    return br, bz, bt, bp


def get_mag_field_original(shot: int, Rin, zin, time: float, flag_MSE = None, **kwargs):
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
            # for ii in range(len(time)):
            #     efit_eq = equilibrium(
            #         shot="/common/uda-scratch/lkogan/efitpp_eshed/epm{:0>6}.nc".
            #         format(shot) if shot < 44000 else shot,
            #         device='MASTU', time=time[ii]
            #     )
            #     for jj in range(len(Rin)):
            #         br[ii, jj] = efit_eq.BR(Rin[jj], zin[jj])
            #         bz[ii, jj] = efit_eq.BZ(Rin[jj], zin[jj])
            #         bp[ii, jj] = efit_eq.Bp(Rin[jj], zin[jj])
            #         bt[ii, jj] = efit_eq.Bt(Rin[jj], zin[jj])
            client = pyuda.Client()
            t = client.get('/epm/time', shot).data
            r = client.get('/epm/output/profiles2d/r', shot).data
            z = client.get('/epm/output/profiles2d/z', shot).data
            for ii in range(len(time)):
                idxt = (np.abs(t-time[ii])).argmin()
                for jj in range(len(Rin)):
                    idxr = (np.abs(r-Rin[jj])).argmin()
                    idxz = (np.abs(z-zin[jj])).argmin()
                    br[ii, jj] = client.get('/epm/output/profiles2d/br', shot).data[idxt,idxr,idxz]
                    bz[ii, jj] = client.get('/epm/output/profiles2d/bz', shot).data[idxt,idxr,idxz]
                    bp[ii, jj] = client.get('/epm/output/profiles2d/bpol', shot).data[idxt,idxr,idxz]
                    bt[ii, jj] = client.get('/epm/output/profiles2d/bphi', shot).data[idxt,idxr,idxz]
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
        # except pyuda.ServerException:
        #     logger.warning('MSE-constrained data BAD or non existent. Will use regular equilibrium')
        #     for ii in range(len(time)):
        #         efit_eq = equilibrium(
        #             shot="/common/uda-scratch/lkogan/efitpp_eshed/epm{:0>6}.nc".
        #             format(shot) if shot < 44000 else shot,
        #             device='MASTU', time=time[ii]
        #         )
        #         for jj in range(len(Rin)):
        #             br[ii, jj] = efit_eq.BR(Rin[jj], zin[jj])
        #             bz[ii, jj] = efit_eq.BZ(Rin[jj], zin[jj])
        #             bp[ii, jj] = efit_eq.Bp(Rin[jj], zin[jj])
        #             bt[ii, jj] = efit_eq.Bt(Rin[jj], zin[jj])
        except pyuda.ServerException:
            logger.warning('MSE-constrained data BAD or non existent. Will use regular equilibrium')
            client = pyuda.Client()
            t = client.get('/epm/time', shot).data
            r = client.get('/epm/output/profiles2d/r', shot).data
            z = client.get('/epm/output/profiles2d/z', shot).data
            for ii in range(len(time)):
                idxt = (np.abs(t-time[ii])).argmin()
                for jj in range(len(Rin)):
                    idxr = (np.abs(r-Rin[jj])).argmin()
                    idxz = (np.abs(z-zin[jj])).argmin()
                    br[ii, jj] = client.get('/epm/output/profiles2d/br', shot).data[idxt,idxr,idxz]
                    bz[ii, jj] = client.get('/epm/output/profiles2d/bz', shot).data[idxt,idxr,idxz]
                    bp[ii, jj] = client.get('/epm/output/profiles2d/bpol', shot).data[idxt,idxr,idxz]
                    bt[ii, jj] = client.get('/epm/output/profiles2d/bphi', shot).data[idxt,idxr,idxz]
    return br, bz, bt, bp
