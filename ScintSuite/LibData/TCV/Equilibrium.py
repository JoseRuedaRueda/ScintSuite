"""Routines for the magnetic equilibrium"""
import numpy as np
from scipy.interpolate import interpn, interp1d
import os
import xarray as xr
import warnings
import scipy
import os, re, getpass
user = getpass.getuser()

try:
    #Make use of eqtools package to read TCV LIQUE equilibrium
    import eqtools
except:
    warnings.warn(
        "eqtools not loaded"
        "reading LIUQE data from MDS will not work."
    )
try:
    from freeqdsk import geqdsk
except:
    warnings.warn(
        "freeqdsk not loaded"
        "reading geqdsk files will not work."
    )

# -----------------------------------------------------------------------------
# --- eqiulibrium data
# -----------------------------------------------------------------------------
def get_mag_field_old(shot: int, Rin: float, zin: float, time: float, **kwargs):
    """
    Get TCV magnetic field

    Jesus Poley-Sanjuan - jesus.poley@epfl.ch
   
    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this

        :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T]

    WARNING: if the br component is negative, the bz sign is not read if it's negative

    """
    Z = zin# vertical position [m]
    R = Rin# radial position [m]
    # Create a temporary file named B.txt to store the local B [Br,Bphi,Bz] at
    a=f"matlab -nodisplay -r \"addpath('/home/poley/FILD/Jesus');" \
    f"eq=gdat({shot},'eqdsk','time',{time});" \
    f"equil=read_eqdsk(eq.eqdsk);" \
    f"R={R};Z={Z};position=1e3*[R*cos(0),R*sin(0),Z];" \
    f"B=extractDataEqdsk(equil,position,'Bcart');" \
    f"save('B.txt','B','-ASCII','-append');exit\""
    os.system(a) # Executing the a string
    # Read B.txt to get B field
    print('Reading magnetic field')
    f = open('B.txt') # Open txt file
    B = f.readlines()[0] # Read txt file containing B [T]
    br = float(B[2:16]) # take characters of Br and convert them to float
    bt = float(B[18:32]) # take characters of Br and convert them to float
    bz = float(B[35:48]) # ake characters of Br and convert them to float
    # Poloidal field is just obtained from Br and Bz
    bp = np.sqrt(br**2 + bz**2) # [T]
    # Delete B.txt file from current directory
    os.remove('B.txt')
    return br, bz, bt, bp

def get_mag_field(shot: int, Rin: float, zin: float, time: float, 
                   **kwargs):
    '''
    Function to get B-field at R, Z position

    Two options to load magnetic euilibrium data are implemetned
    1) using the gdat function available in matlab at TCV, a CHEASE equilibrium is calculated
    and stored in an eqdsk file which is then read using the freeqdsk package.
    CHEASE uses a LIUQE equilibrium as input, and calculates an equilibrium with the LIUQE boundary fixed
    Since Matlab needs to be used to call gdat and the CHEASE calculation takes several seconds, this option is slower
    But the CHEASE equilibriumshould have a more realistic internal flux distribution. 
    Note however, that the kinetic profiles from conf are typically mapped to LIUQE.
    
    2) using the eqtools package the LIUQE.m eqiulibrium can be read.
    This is much faster. But results will differ slightly from the CHEASE equilibrium

    The uncertainty in magnetic field outside the LCFS is large for both options.

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch
    Jesus Poley-Sanjuan - jesus.poley@epfl.ch
    
    Note: No extra arguments are expected, **kwargs is just included for
    compatibility of the call to this method in other databases (machines)

    :param  shot: Shot number
    :param  Rin: Array of R positions where to evaluate (in pairs with zin) [m]
    :param  zin: Array of z positions where to evaluate (in pairs with Rin) [m]
    :param  time: Array of times where we want to calculate the field (the
    field would be calculated in a time as close as possible to this

    Optional: make use of **kwargs
    :param   use_gdat: If set to True, gdat will be used to write a CHEASE eqdsk file 
                        which is then read to get the magnetic field
                       if False (default), the LUIQE.m functions will be used to read
                        the LUIQE equilbrium and used to get the magnetic field
    :param   liuqe: (int) can be 1, 2, 3. Standard is 1


    :return br: Radial magnetic field (nt, nrz_in), [T]
    :return bz: z magnetic field (nt, nrz_in), [T]
    :return bt: toroidal magnetic field (nt, nrz_in), [T]
    :return bp: poloidal magnetic field (nt, nrz_in), [T] 

    '''

    default_options = {
        'liuqe': 1,
        'use_gdat': False
    }
    default_options.update(kwargs)

    if default_options['use_gdat']:
        cmd = f"matlab -nodisplay -r \"eq=gdat({shot},'eqdsk','time',{time},'liuqe',{default_options['liuqe']},'source','chease');exit\"" #important no spaces allowed in LAC10
        os.system(cmd)
        user = getpass.getuser()
        fn_eqdsk = f'/tmp/{user}/EQDSK_{shot}t{time:1.4f}_COCOS17'
        
        with open(fn_eqdsk, "r") as f:
            eqd = geqdsk.read(f)

        b2d = {
            "nr" : eqd["nx"],
            "rmin" : eqd["rleft"], "rmax" : eqd["rleft"]+eqd["rdim"],
            "nz" : eqd["ny"],
            "zmin" : eqd["zmid"] - 0.5*eqd["zdim"],
            "zmax" : eqd["zmid"] + 0.5*eqd["zdim"],
            "axisr" : eqd["rmagx"], "axisz" : eqd["zmagx"],
            "psi" : eqd["psi"], "psi0" : eqd["simagx"], "psi1" : eqd["sibdry"],
            "br" : eqd["psi"]*0, "bz" : eqd["psi"]*0
        }

        rgrid = np.linspace(b2d['rmin'], b2d['rmax'], b2d['nr'])
        zgrid = np.linspace(b2d['zmin'], b2d['zmax'], b2d['nz'])
        rmesh, zmesh   = np.meshgrid(rgrid, zgrid, indexing="ij")

        psiOfRZSpline = scipy.interpolate.RectBivariateSpline(
            rgrid,
            zgrid,
            b2d["psi"]/ (2.0 * scipy.pi),
            s=0
        )

        _currentSign = 1 if np.mean(eqd['cpasma']) > 1e5 else -1

        b2d['br'] = - 1.0 / rmesh * psiOfRZSpline.ev(rmesh, zmesh, dx=0, dy=1)  * -1.0 * _currentSign
        b2d['bz'] = 1.0 / rmesh * psiOfRZSpline.ev(rmesh, zmesh, dx=1, dy=0)  * -1.0 * _currentSign

        # Toroidal component is more complicated for it can be evaluated from
        # Btor = F/R but we need to map F(psi) to F(R,z) first. However, F(psi)
        # is given only inside the plasma.
        psigrid = np.linspace(eqd["simagx"], eqd["sibdry"], eqd["nx"])
        if eqd["simagx"] < eqd["sibdry"]:
            fpolrz  = np.interp(eqd["psi"], psigrid, eqd["fpol"],
                                right=eqd["fpol"][-1])
        else:
            fpolrz  = np.interp(eqd["psi"], psigrid[::-1], eqd["fpol"][::-1],
                                right=eqd["fpol"][-1])

        if eqd["nx"] != b2d["nr"]: fpolrz = fpolrz[1:,:] # If we had rmin=0
        b2d["bphi"] = fpolrz/rmesh

        br = interpn((rgrid, zgrid), b2d['br'], (Rin, zin), fill_value=0.0, method='cubic')
        bz = interpn((rgrid, zgrid), b2d['bz'], (Rin, zin), fill_value=0.0, method='cubic')
        bt = interpn((rgrid, zgrid), b2d['bphi'], (Rin, zin), fill_value=0.0, method='cubic')

        # Poloidal field is just obtained from Br and Bz
        bp = np.hypot(br, bz)

    else:
        eq = eqtools.TCVLIUQEMATTree(shot)

        rGrid = eq.getRGrid()
        zGrid = eq.getZGrid()
        meshR, meshZ = np.meshgrid(rGrid, zGrid, indexing = 'ij')

        psi = eq.getFluxGrid()
        psi0, psi1 = eq.getFluxAxis(), eq.getFluxLCFS()

        br = eq.rz2BR(meshR, meshZ, time)  
        #bphi = eq.rz2BT(meshR.T, meshZ.T, time) # This only works inside the LCFS, so isntead follow example in importData from a5py
        bz, it = eq.rz2BZ(meshR, meshZ, time, return_t = True)

        br = br * -1.0 * eq.getCurrentSign()  #TCV current sign convention differes form eqtools convention
        bz = bz * -1.0 * eq.getCurrentSign()

        it = it[0][0]

        # Toroidal component is more complicated for it can be evaluated from
        # Btor = F/R but we need to map F(psi) to F(R,z) first. However, F(psi)
        # is given only inside the plasma.

        ##BT is tricky
        F = eq.getF()
        psigrid = np.linspace(psi0[it], psi1[it], np.shape(F[it])[0]).T

        if psi0[it] < psi1[it]:
            fpolrz  = np.interp(psi[it].T, psigrid, F[it], right=F[it, -1])
        else:
            fpolrz  = np.interp(psi[it].T, psigrid[::-1], F[it,::-1], right=F[it,-1])
        bphi = fpolrz / meshR

        br = interpn((rGrid, zGrid), br, (Rin, zin), fill_value=0.0, method='cubic')
        bz = interpn((rGrid, zGrid), bz, (Rin, zin), fill_value=0.0, method='cubic')
        bt = interpn((rGrid, zGrid), bphi, (Rin, zin), fill_value=0.0, method='cubic')

        # Poloidal field is just obtained from Br and Bz
        bp = np.hypot(br, bz)

    return br, bz, bt, bp

# -----------------------------------------------------------------------------
# --- q_profile
# -----------------------------------------------------------------------------
def get_q_profile(shotnumber: int = None, time: float = None,
                  diag: str = 'LIUQE',
                  **kwargs):
    """
    Wrapper to read q-profile either from LIUQE data on MDS or calculate using CHEASE.

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    :param  shot (int): Shot number
    :param  diag (str): Option between 'LIUQE' or 'CHEASE' from which to get q profile
    :param  time: Array of times where we want to calculate the field
    kwargs:
        liuqe (int): version of LIUQE to use, default is 1
        xArrayOutput (bool): if set to True return data in xarray format, 
                            if False, return data in dictionary format
                            default is True
    :return
    output: dictionary or xarray(default)
        timebase [nt]
        q_profile [nt, nrho]
        rho [nt, nrho]
    """

    default_options = {
        'liuqe': 1,
        'xArrayOutput': True
    }
    default_options.update(kwargs)


    if diag not in ('LIUQE', 'CHEASE'):
        raise Exception('Diagnostic non supported!')

    if diag == 'LIUQE':
        return get_q_profile_LIUQE(shot = shotnumber, time = time, 
                                   **default_options)
    elif diag == 'CHEASE':
        return get_q_profile_CHEASE(shot = shotnumber, time = time, 
                                   **default_options)

def get_q_profile_LIUQE(shot: int, 
                  liuqe: int = 1, time: float = None, 
                  xArrayOutput: bool = True,
                  **kwargs):
    """
    Reads LIUQE calculated q-profile from MDS.

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    :param  shot (int): Shot number
    :param  liuqe (int): version of LIUQE to use
    :param  time: Array of times where we want to calculate the field

    :return

    timebase [nt]
    q_profile [nt, nrho]
    rho [nt, nrho]
    """
    default_options = {
        'q_absolute': False
    }
    default_options.update(kwargs)

    eq = eqtools.TCVLIUQEMATTree(shot)
    eq.getTimeBase()
    timebasis =eq._time
    eq.getRhoProfile()
    rho = eq._rhopsi
    eq.getQProfile()
    qpsi = eq._qpsi

    if default_options['q_absolute']:
        qpsi = np.abs(qpsi.T)


    if time is not None:
        time = np.atleast_1d(time)

    if not xArrayOutput:
        if time is None:
            output = {
                'data': qpsi,
                'time': timebasis,
                'rho': rho
            }

        elif len(time) >= 1:
            output = {
                'data': interp1d(timebasis, qpsi, axis=0)(time).squeeze(),
                'time': time.squeeze(),
                'rho': interp1d(timebasis, rho, axis=0)(time).squeeze()
            }

        output['source'] = {
            'liuqe': liuqe,
            'pulseNumber': shot
        }
    else:
        output = xr.Dataset()

        output['data'] = xr.DataArray(qpsi.T, dims=('rho', 't'),
                                      coords={'rho': rho[0, :],
                                      't': timebasis})
        output['data'].attrs['long_name'] = 'q'
        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'

        output.attrs['liuqe'] = liuqe
        output.attrs['shot'] = shot


    return output

def get_q_profile_CHEASE(shot: int, 
                  liuqe: int = 1, time: float = None, 
                  xArrayOutput: bool = True):
    """
    Runs CHEASE via gdat and reads q-profile from eqdsk file.

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    :param  shot (int): Shot number
    :param  liuqe (int): version of LIUQE to use
    :param  time: Array of times where we want to calculate the field

    :return
    """

    cmd = f"matlab -nodisplay -r \"eq=gdat({shot},'eqdsk','time',{time},'liuqe',{liuqe},'source','chease');exit\"" #important no spaces allowed in LAC10
    os.system(cmd)
    user = getpass.getuser()
    fn_eqdsk = f'/tmp/{user}/EQDSK_{shot}t{time:1.4f}_COCOS17'
    
    with open(fn_eqdsk, "r") as f:
        eqd = geqdsk.read(f)

    psigrid = np.linspace(eqd["simagx"], eqd["sibdry"], eqd["nx"])

    rho = np.sqrt(1 - psigrid/psigrid[0])
    qpsi = eqd['qpsi']

    if not xArrayOutput:
        output = {
            'data': qpsi,
            'time': time,
            'rho': rho
        }
    else:
        output = xr.Dataset()

        output['data'] = xr.DataArray(qpsi, dims=('rho'),
                                      coords={'rho': rho,
                                      't': time})
        output['data'].attrs['long_name'] = 'q'
        output['rho'].attrs['long_name'] = '$\\rho_p$'
        output['t'].attrs['long_name'] = 'Time'
        output['t'].attrs['units'] = 's'

        output.attrs['liuqe'] = liuqe
        output.attrs['shot'] = shot

    return output

# -----------------------------------------------------------------------------
# --- time traces of global parameters
# -----------------------------------------------------------------------------

def get_shot_basics(shotnumber: int,
                    time: float = None):
    """
    Retrieves from the LIUQE.m equilibrium reconstruction and 
    returns some of the basic global parameters in a dictionary structure

    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    :param  shot: Shot number.
    :param  time: time interval to retrieve. If it is a single value, only the
    appropriate data point will be retrieved. If None, all the data points are
    obtained.
    """

    eq = eqtools.TCVLIUQEMATTree(shotnumber)
    eq.getTimeBase()
    timebasis =eq._time

    nt = len(timebasis)
    if time is None:
        t0 = 0
        t1 = nt
    elif len(time) == 1:
        t0 = np.abs(timebasis.flatten() - time).argmin()
        t1 = t0+1
    else:
        t0 = np.abs(timebasis.flatten() - time[0]).argmin() - 1
        t1 = np.abs(timebasis.flatten() - time[-1]).argmin() + 1

    bt0 = eq.getBCentr()  
    Rmag = eq.getMagR()
    ahor = eq.getAOut()
    kappa = eq.getElongation()

    out = {
        'time': np.atleast_1d(timebasis[t0:t1]),
        'bt0': bt0[t0:t1],
        'bttime': np.atleast_1d(timebasis[t0:t1]), # this extra time base is because of the AUG implemetation spilling over into _MHD
        'Rmag': Rmag[t0:t1],
        'ahor': ahor[t0:t1] ,
        'k': kappa[t0:t1]
    }

    return out