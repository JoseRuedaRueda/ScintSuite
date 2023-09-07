"""Routines for the magnetic equilibrium"""
import numpy as np
from pyEquilibrium.equilibrium import equilibrium


def get_mag_field(shot: int, Rin, zin, time: float, **kwargs):
    """
    Get TCV magnetic field

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
    Jesus Poley-Sanjuan - jesus.poley@epfl.ch

    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)

    Note2: MU FILD1 is located around z=0.159m

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this

    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]
    """
    if isinstance(time, (list, np.ndarray)):
        pass
    else:  # it should be just a number
        time = np.array([time])
    br = np.zeros(time.shape)
    bz = np.zeros(time.shape)
    bp = np.zeros(time.shape)
    bt = np.zeros(time.shape)

    for ii in range(len(time)):
        efit_eq = equilibrium(
            shot="/common/uda-scratch/lkogan/efitpp_eshed/epm{:0>6}.nc".
            format(shot) if shot < 44849 else shot,
            device='MASTU', time=time[ii]
        )
        br[ii] = efit_eq.BR(Rin, zin)
        bz[ii] = efit_eq.BZ(Rin, zin)
        bp[ii] = efit_eq.Bp(Rin, zin)
        bt[ii] = efit_eq.Bt(Rin, zin)

    return br, bz, bt, bp
