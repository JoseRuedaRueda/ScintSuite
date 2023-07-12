"""
Read FIDASIM Outputs

Base created by Plar Cano: pcano@us.es and ajvv. Adapted and extended by
Jose Rueda: jrrueda@us.es

Python reading routines for FIDASIM output

Contains:
    - read_neutrals
    - read_spec
    - read_grid
Still to implement: weighting functions, Zeeman lines
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import ScintSuite._Parameters as sspar


def read_neutrals(filename: str):
    """
    Read the NBI and Halo neutrals

    Plar Cano: pcano@us.es and Jose Rueda: jrrueda@us.es

    :param  filename: name of the neutral file to read

    :return data: Dictionary containing:
        - shot: shot number
        - time: time point [s]
        - nx, ny, nz: grid size
        - nlevs: number of spectra levels
        - fdens: density of 1 energy neutrals (nl, nz, ny, nx)
        - hdens: density of 2nd energy neutrals (nl, nz, ny, nx)
        - tdens: density of 3rd energy neutrals (nl, nz, ny, nx)
        - halodens: density of halo neutrals (nl, nz, ny, nx)
        - dcxdens: [optional] density of direct CX neutrals (nl, nz, ny, nx)
    """
    float_type = np.single

    with open(filename, 'rb') as fh:
        data = {
            'shot': int(np.fromfile(fh, dtype=float_type, count=1)),
            'time': float(np.fromfile(fh, dtype=float_type, count=1)),
            'nx': int(np.fromfile(fh, dtype=float_type, count=1)),
            'ny': int(np.fromfile(fh, dtype=float_type, count=1)),
            'nz': int(np.fromfile(fh, dtype=float_type, count=1)),
            'nlevs': int(np.fromfile(fh, dtype=float_type, count=1)),
        }
        (nx, ny, nz, nl) = (data['nx'], data['ny'], data['nz'], data['nlevs'])
        dim = nx*ny*nz*nl

        data['fdens'] = np.fromfile(fh, dtype=float_type,
                                    count=dim).reshape(nl, nz, ny, nx)
        data['hdens'] = np.fromfile(fh, dtype=float_type,
                                    count=dim).reshape(nl, nz, ny, nx)
        data['tdens'] = np.fromfile(fh, dtype=float_type,
                                    count=dim).reshape(nl, nz, ny, nx)
        data['halodens'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nl, nz, ny, nx)
        try:
            data['dcxdens'] = np.fromfile(fh, dtype=float_type,
                                          count=dim).reshape(nl, nz, ny, nx)
        except EOFError:
            print('dcxdens not written in the file')
            pass
        except ValueError:
            print('dcxdens not written in the file')
            pass
    return data


def read_spec(filename: str, spectra_stark_resolved: bool = True):
    """
    Read the spectral lines

    Plar Cano: pcano@us.es

    :param  filename: name of the file with FIDASIM the results
    :param  spectra_stark_resolved: If there stark resolved data was simulated
    """
    data = {}
    int_type = np.int32
    float_type = np.single
    long_type = np.single
    with open(filename, 'rb') as fh:
        dumm = np.fromfile(fh, dtype=int_type, count=1)[0]  # shot
        data['shot'] = dumm
        print('Shot number:', dumm)
        nstark = np.fromfile(fh, dtype=long_type, count=1)[0]
        if nstark == 15:
            data['nstark'] = 15
            nstark = 15.0
        else:
            data['nstark'] = 2
            nstark = 2.0
        data['nlos'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        data['nlambda'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        (ns, nla, nlo) = (data['nstark'], data['nlambda'], data['nlos'])
        data['lambda'] = np.fromfile(fh, dtype=float_type, count=nla)
        if spectra_stark_resolved:
            dim = int(nstark*nla*nlo)
            data['full'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla, ns)
            data['half'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla, ns)
            data['third'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla, ns)
            data['halo'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla, ns)
            try:
                data['dcx'] = np.fromfile(fh, dtype=float_type,
                                          count=dim).reshape(nlo, nla, ns)
            except EOFError:
                pass
            except ValueError:
                pass
        else:
            dim = nla*nlo
            data['full'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla)
            data['half'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla)
            data['third'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla)
            data['halo'] = np.fromfile(fh, dtype=float_type,
                                       count=dim).reshape(nlo, nla)
            try:
                data['dcx'] = np.fromfile(fh, dtype=float_type,
                                          count=dim).reshape(nlo, nla)
            except EOFError:
                pass
            
            except ValueError:
                pass
    try:
        folder, file = os.path.split(filename)
        rhofile = os.path.join(folder, 'rhodiag.bin')
        data['rho_diag'] = read_rho_diag(rhofile)
    except:
        pass
    return data


def read_grid(filename: str):
    """
    Read the FIDASIM grid

    Plar Cano: pcano@us.es and Jose Rueda: jrrueda@us.es

    :param  filename: name of the file with the FIDASIM4 grid

    :return grid: used grid in FIDASIM
    """
    grid = {}
    int_type = np.int32
    double_type = np.float64
    print('Reading: ', filename)
    with open(filename, 'rb') as fh:
        grid['nx'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        grid['dx'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['xmin'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['xmax'] = np.fromfile(fh, dtype=double_type, count=1)[0]

        grid['ny'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        grid['dy'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['ymin'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['ymax'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['nz'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        grid['dz'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['zmin'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['zmax'] = np.fromfile(fh, dtype=double_type, count=1)[0]

        grid['nr'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        grid['dr'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['rmin'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['rmax'] = np.fromfile(fh, dtype=double_type, count=1)[0]

        grid['dv'] = np.fromfile(fh, dtype=double_type, count=1)[0]
        grid['ntrack'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        grid['dl'] = np.fromfile(fh, dtype=double_type, count=1)[0]

        grid['xx'] = np.fromfile(fh, dtype=double_type, count=grid['nx'])
        grid['yy'] = np.fromfile(fh, dtype=double_type, count=grid['ny'])
        grid['zz'] = np.fromfile(fh, dtype=double_type, count=grid['nz'])
        grid['rr'] = np.fromfile(fh, dtype=double_type, count=grid['nr'])

    grid['xaxis'] = np.linspace(grid['xmin'], grid['xmax'], grid['nx'])
    grid['yaxis'] = np.linspace(grid['ymin'], grid['ymax'], grid['ny'])
    grid['zaxis'] = np.linspace(grid['zmin'], grid['zmax'], grid['nz'])
    return grid


def read_profiles(filename: str):
    """
    Read used profiles

    :param  filename: mane of the profile fidasim file

    :return profile dict containing:
        - nrho: number of poins of the grid
        - drho: rho spacing
        - rho: rho axis
        - te, ti: electron and ion temperatures [] (nrho)
        - dene: electron density [] (nrho)
        - denp: main ion density [] (nrho)
        - deni: impurity density [] (nrho)
        - vtor: toroidal rotation [] (nrho)
        - zeff: Zeff [] (nrho)
        - background_dens: bck neutral density [] (nrho)
        - rho_str: string indicating if rho is toroidal or poloidal

    """
    int_type = np.int32
    double_type = np.float64
    str_type = '|S5'

    with open(filename, 'rb') as fh:
        nrho = np.fromfile(fh, dtype=int_type, count=1)[0]
        profiles = {
            'nrho': nrho,
            'drho': np.fromfile(fh, dtype=double_type, count=1)[0],
            'rho': np.fromfile(fh, dtype=double_type, count=nrho),
            'te': np.fromfile(fh, dtype=double_type, count=nrho),
            'ti': np.fromfile(fh, dtype=double_type, count=nrho),
            'dene': np.fromfile(fh, dtype=double_type, count=nrho),
            'denp': np.fromfile(fh, dtype=double_type, count=nrho),
            'deni': np.fromfile(fh, dtype=double_type, count=nrho),
            'vtor': np.fromfile(fh, dtype=double_type, count=nrho),
            'zeff': np.fromfile(fh, dtype=double_type, count=nrho),
            'background_dens': np.fromfile(fh, dtype=double_type, count=nrho),
            'rho_str': np.fromfile(fh, dtype=str_type, count=1)[0].decode()
        }
    return profiles


def read_fida(filename: str, spectra_stark_resolved: bool = True,
              nstark: int = -1):
    """
    Read FIDA simulated signal

    :param  filename: name of the folder with FIDASIM the results
    """
    data = {}
    int_type = np.int32
    float_type = np.single
    if nstark == -1:
        spec = read_spec(filename)
        ns = spec['nstark']
    with open('%s/fida_spectra.bin' % (filename), 'rb') as fh:
        np.fromfile(fh, dtype=int_type, count=1)[0]
        np.fromfile(fh, dtype=float_type, count=1)[0]
        data['nlos'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        data['nlambda'] = np.fromfile(fh, dtype=int_type, count=1)[0]
        data['lambda'] = np.fromfile(fh, dtype=float_type,
                                     count=data['nlambda'])
        (nla, nlo) = (data['nlambda'], data['nlos'])

        if spectra_stark_resolved:
            dim = ns*nla*nlo
            data['afida'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla, ns)
            data['pfida'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla, ns)
            try:
                data['pthermal'] = np.fromfile(fh, dtype=float_type,
                                               count=dim).reshape(nlo, nla, ns)
            except EOFError:
                pass
        else:
            dim = nla*nlo
            data['afida'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla)
            data['pfida'] = np.fromfile(fh, dtype=float_type,
                                        count=dim).reshape(nlo, nla)
            try:
                data['pthermal'] = np.fromfile(fh, dtype=float_type,
                                               count=dim).reshape(nlo, nla)
            except EOFError:
                pass

    return data


def read_rho_diag(filename):
    """
    Read rho coorditantes

    Plar Cano: pcano@us.es and Jose Rueda: jrrueda@us.es

    :param  filename: name of the file with the rho coordinates

    :return data: dictionary containing the coordinates
        - dim: dimenssion of the grid
        - rhot: rho toroidal
        - rhop: rho poloidal
    """
    data = {}
    int_type = np.int32
    double_type = np.float64
    with open(filename, 'rb') as fh:
        dim = int(np.fromfile(fh, dtype=int_type, count=1)[0])
        data = {
            'dim': dim,
            'rhop': np.fromfile(fh, dtype=double_type, count=dim),
            'rhot': np.fromfile(fh, dtype=double_type, count=dim),
        }
    return data


def read_field(filename, nr: int = None, nz: int = None):
    """
    Read the electromagnetic field

    Plar Cano: pcano@us.es and Jose Rueda: jrrueda@us.es

    :param  filename: name of the field file
    :param  nr: number of grid points in r direction
    :param  nz: number of grid points in z direction

    :return: dict containing:
        - brzt: field Br, Bz, Bt [T], (3, nr, nz)
        - efield: electric field Er, Ez, Et [V/cm]
        - rho_grid: rho grid
        - grid: spatial grid, only if nr or nz were None

    Note: Only works with 2D fields!
    Note: if nr or nz are None, the grid saved in the same folder will be load
    and used
    """
    double_type = np.float64
    readed_grid = False
    if (nr is None) or (nz is None):
        folder, file = os.path.split(filename)
        grid = read_grid(os.path.join(folder, 'grid.bin'))
        (nr, nz) = (grid['nr'], grid['nz'])
        readed_grid = True
    dim = nr*nz*3

    with open(filename, 'rb') as fh:
        data = {
            'brzt': np.fromfile(fh, dtype=double_type,
                                count=dim).reshape(3, nz, nr),
            'erzt': np.fromfile(fh, dtype=double_type,
                                count=dim).reshape(3, nz, nr),
            'rho_grid': np.fromfile(fh, dtype=double_type,
                                    count=nr*nz).reshape(nz, nr)
        }
    if readed_grid:
        data['grid'] = grid
    return data


def read_diag(filename):
    """
    Read the diagnostic

    Plar Cano: pcano@us.es and Jose Rueda: jrrueda@us.es

    :param  filename: name of the file with the fidasim diagnostic

    Note: still have done, weights of LOS not read, neither alpha and beta for
        the INPA
    """
    data = {}
    int_type = np.int32
    double_type = np.float64
    with open(filename, 'rb') as fh:
        nchan = np.fromfile(fh, dtype=int_type, count=1)[0]
        data['nchan'] = nchan
        data['xyzhead'] = np.fromfile(fh, dtype=double_type,
                                      count=nchan*3).reshape(3, nchan)
        data['xyzlos'] = np.fromfile(fh, dtype=double_type,
                                     count=nchan*3).reshape(3, nchan)
        data['headsize'] = np.fromfile(fh, dtype=double_type,
                                       count=nchan)
        data['opening_angle'] = np.fromfile(fh, dtype=double_type,
                                            count=nchan)
        data['sigma_pi'] = np.fromfile(fh, dtype=double_type,
                                       count=nchan)
        data['instfu'] = np.fromfile(fh, dtype=double_type,
                                     count=nchan)
        data['xhead'] = data['xyzhead'][0, :]
        data['yhead'] = data['xyzhead'][1, :]
        data['zhead'] = data['xyzhead'][2, :]
        data['xlos'] = data['xyzlos'][0, :]
        data['ylos'] = data['xyzlos'][1, :]
        data['zlos'] = data['xyzlos'][2, :]
        # data['los_name'] = np.fromfile(fh,dtype=double_type,count=nchan)
        # @todo Need to implement weight
        # @todo Need to implement alpha y beta for INPA
    fh.close()
    return data


def read_spac_res(filename, nr=-1, nz=-1, nchan=-1, stell_ind=0.0):
    """
    Read spac res

    Plar Cano: pcano@us.es

    :param  filename: name of the folder with FIDASIM the results
    """
    data = {}

    float_type = np.float64
    if nr == -1 or nz == -1:
        grid = read_grid(filename)
        (nr, nz) = (grid['nr'], grid['nz'])
    if nchan == -1:
        diag = read_diag(filename)
        nchan = diag['nchan']
    with open(filename, 'rb') as fh:
        data['spac_res'] = \
            np.fromfile(fh, dtype=float_type,
                        count=nr*nz*nchan).reshape(nchan, nz, nr)
    fh.close()
    return data


def read_bremmstrahlung(filename, lambda_in=np.asarray([0]), diag=None,
                        field=None, settings=None, grid=None, prof=None,
                        stell_ind=None, doplt=False):
    """
    Read Bremmstrahlung data

    Plar Cano: pcano@us.es

    :param  filename: file with FIDASIM results
    """
    data = {}
    h_planck = sspar.h_planck
    c0 = sspar.c
    folder, file = os.path.split(filename)
    if grid is None:
        grid = read_grid(os.path.join(folder, 'grid.bin'))
    if diag is None:
        diag = read_diag(os.path.join(folder, 'diag.bin'))
    if settings is None:
        settings = read_settings(os.path.join(folder, 'namelist.dat'))
    if prof is None:
        prof = read_profiles(os.path.join(folder, 'profiles.bin'))
    if lambda_in[0] == 0:
        spectra = read_spec(os.path.join(folder, 'nbi_halo_spectra.bin'))
        lambda_in = spectra['lambda']
    if field is None:
        field = read_field(os.path.join(folder, 'field.bin'))

    wavel = lambda_in * 1.e-9
    dl = grid['dl'] / 2.   # step length
    nstep = np.int(1000.0 / dl)  # (follow LOS maximal 10 m)
    brems = np.zeros([settings['nlambda'], diag['nchan']])
    xyz_pos = np.zeros([nstep, 3])
    for ichan in range(diag['nchan']):
        xyzhead = diag['xyzhead'][:, ichan]
        xyzlos = diag['xyzlos'][:, ichan]
        # determine positions along LOS
        vi = xyzlos-xyzhead
        vi /= np.sqrt(np.sum(vi**2))  # unit vector
        # dummy = np.max(np.abs(vi))
        ic = np.argmax(np.abs(vi))
        dpos = vi/np.abs(vi[ic])
        dpos = dpos/np.sqrt(np.sum(dpos**2)) * dl
        xyz_pos[0, :] = xyzhead

        steps = np.arange(nstep-1)+1
        for ii in steps:
            xyz_pos[ii, :] = xyz_pos[ii-1, :] + dpos
        rpos = np.sqrt(xyz_pos[:, 0]**2+xyz_pos[:, 1]**2)

        phipos = np.arctan(xyz_pos[:, 1], xyz_pos[:, 0])
        # limit the positions to the boundaries of the r-grid!
        index = np.where(rpos < np.max(grid['rr']))[0]
        rpos = rpos[index]
        zpos = xyz_pos[index, 2]
        phipos = phipos[index]

        # get rho values of grid
        rhoTF = np.zeros(len(rpos))
        if stell_ind:
            for ii in range(len(rpos)):
                if phipos[ii] < 0:
                    phipos[ii] += 2. * np.pi
                nsym = np.round(np.pi/(field['dphi']*(field['nphi']-1)))

                while phipos[ii] > field['dphi']*(field['nphi']-1):
                    phipos[ii] -= 2. * np.pi / nsym
                if phipos[ii] < 0:
                    phipos[ii] *= -1.
                rax = np.arange(field['nr'])*field['dr'] + field['rmin']
                zax = np.arange(field['nz'])*field['dz'] + field['zmin']
                phax = np.arange(field['nphi'])*field['dphi'] + field['phimin']
                indr = np.argmin(np.abs(rpos[ii]-rax))
                indz = np.argmin(np.abs(zpos[ii]-zax))
                indphi = np.argmin(np.abs(phipos[ii]-phax))
                rhoTF[ii] = field['rho_grid'][indr, indz, indphi]

        else:
            from scipy.interpolate import interp2d
            f = interp2d(grid['rr'], grid['zz'], field['rho_grid'])
            for i in range(len(rpos)):
                rhoTF[i] = f(rpos[i], zpos[i])[0]
        # Plot if needed
        if doplt:
            plt.plot(rpos, zpos)
            rmin = grid['rmin']
            rmax = grid['rmax']
            zmin = grid['zmin']
            zmax = grid['zmax']
            clr = 'k'
            alpha = 1
            plt.plot([rmin, rmin], [zmin, zmax], color=clr, alpha=alpha)
            plt.plot([rmax, rmax], [zmin, zmax], color=clr, alpha=alpha)
            plt.plot([rmin, rmax], [zmin, zmin], color=clr, alpha=alpha)
            plt.plot([rmin, rmax], [zmax, zmax], color=clr, alpha=alpha)
            plt.show()
        # set rho to 10 outside the defined grid!
        index = \
            np.concatenate([np.where(rpos < grid['rmin'])[0][:],
                            np.where(rpos > grid['rmax'])[0][:],
                            np.where(zpos < grid['zmin'])[0][:],
                            np.where(zpos > grid['zmax'])[0][:]])

        rhoTF[index] = 10.

        # remove points outside rhotf=1.2!!
        index = np.where(rhoTF < 1.2)[0]
        rhoTF = rhoTF[index]

        # get kintic profiles
        dene = np.interp(rhoTF, prof['rho'], prof['dene']) * 1e-13  # [e19/m^3]
        nind = len(rhoTF)
        if dene[0] > 0.7 or dene[-1] > 0.7:
            line = 'rz-grid too small for good Bremsstrahlung!'\
                + 'set it to zero for this channel!'
            print(line)
            continue
        te = np.interp(rhoTF, prof['rho'], prof['te'])*1e3     # [eV]
        indte = np.where(te < 1.0)[0]
        te[indte] = 1.0
        zeff = np.interp(rhoTF, prof['rho'], prof['zeff'])

        print('LOS length through plasma: %.2f' % (nind*dl))
        for il in range(settings['nlambda']):
            B = 3.108 - np.log(te * 1.0e-3)
            A = 7.57e-7 * dene**2\
                / (wavel[il]*np.sqrt(te))*np.exp(-h_planck*c0/(wavel[il]*te))
            # sort out nans and infinits...
            index = np.where(np.isnan(A*B))[0]
            A[index] = 0.
            B[index] = 0.
            gaunt = 5.542 - B * (0.6905-0.1323 / zeff)\
                + 0.35 * (wavel[il] * 1e9-425.0) / 425.0 * zeff**(1. / zeff)
            brems[il, ichan] = np.sum(A * gaunt * zeff) * dl / 100.0
            # (ph/s-m2-nm-sr)

    data['brems'] = brems.T * 1.e15
    return data


def read_settings(filename: str):
    """
    Read FIDASIM settings

    Plar Cano: pcano@us.es

    :param  filename: name of the FIDASIM namelist file
    """
    data = {}
    f = open(filename, 'r')
    f.readline()
    data['path'] = f.readline().split('\n')[0]
    data['device'] = f.readline().split('\n')[0]
    data['shot'] = np.int(f.readline().split(' ')[1])
    data['time'] = np.float(f.readline().split(' ')[1])
    data['nsource'] = np.int(f.readline().split(' ')[1])
    data['source_arr'] = np.zeros(data['nsource']).astype(np.int)
    for i in range(data['nsource']):
        data['source_arr'][i] = \
            np.int(f.readline().split(' ')[1].split('\n')[0])
    data['transp_runid'] = f.readline().split(' ')[0]
    data['fidasim_runid'] = f.readline().split('\n')[0]
    data['diag'] = f.readline().split('#')[0].strip()
    data['calc_birth'] = np.int(f.readline().split('#')[0])
    data['simfa'] = np.int(f.readline().split('#')[0])
    data['calc_spec'] = np.int(f.readline().split('#')[0])
    data['nr_thermal'] = np.int(f.readline().split('#')[0])
    data['npa'] = np.int(f.readline().split('#')[0])
    data['write_neutrals'] = np.int(f.readline().split('#')[0])
    data['background_dens'] = np.int(f.readline().split('#')[0])
    data['n_fida_iterations'] = np.int(f.readline().split('#')[0])
    data['calc_wght'] = np.int(f.readline().split('#')[0])
    f.readline()
    data['w_switch'] = np.int(f.readline().split('switch')[0])
    data['nr_energy_wght'] = np.int(f.readline().split('#')[0])
    data['nr_pitch_wght'] = np.int(f.readline().split('#')[0])
    data['nr_gyro_wght'] = np.int(f.readline().split('#')[0])
    data['nchan_wght'] = np.int(f.readline().split('#')[0])
    data['emax_wght'] = np.float(f.readline().split('#')[0])
    data['dwav_wght'] = np.float(f.readline().split('#')[0])
    data['wavel_start_wght'] = np.float(f.readline().split('#')[0])
    data['wavel_end_wght'] = np.float(f.readline().split('#')[0])
    data['spectra_stark_resolved'] = np.int(f.readline().split('#')[0])
    data['nr_fast'] = np.int(f.readline().split('#')[0])
    data['nr_ndmc'] = np.int(f.readline().split('#')[0])
    data['nr_halo'] = np.int(f.readline().split('#')[0])
    data['nvelocities'] = np.int(f.readline().split('#')[0])
    data['impurity'] = np.int(f.readline().split('#')[0])
    f.readline()
    data['transp_cdf'] = f.readline().split('\n')[0]
    f.readline()
    data['btipsign'] = np.int(f.readline().split('#')[0])
    f.readline()
    data['ai'] = np.float(f.readline().split('#')[0])
    f.readline()
    data['nlambda'] = np.int(f.readline().split('#')[0])
    data['lambdamin'] = np.float(f.readline().split('#')[0])
    data['lambdamax'] = np.float(f.readline().split('#')[0])
    f.readline()
    data['rotate'] = np.float(f.readline().split('#')[0])
    f.readline()
    f.readline()
    data['transp_exp'] = f.readline().split('#')[0].strip()
    data['transp_shot'] = np.int(f.readline().split('#')[0])
    data['transp_ed'] = np.int(f.readline().split('#')[0])
    f.readline()
    data['parametrisation'] = np.zeros(6)
    for i in range(6):
        data['parametrisation'][i] = np.float(f.readline().split('#')[0])
    f.close()
    return data


def read_npa(filename, verbose: bool = True):
    """
    Read (I)NPA output files.

    Jose Rueda: jrrueda@us.es

    :param  filenam: name of the file to be read
    """
    if verbose:
        print('Reading file: ', filename)
    with open(filename, 'rb') as fid:
        shot = int(np.fromfile(fid, 'int32', 1)[:])
        time = float(np.fromfile(fid, 'float32', 1)[:])
        nr_npa = int(np.fromfile(fid, 'int32', 1)[:])
        counter = int(np.fromfile(fid, 'int32', 1)[:])
        data = {
            'shot': shot,
            'time':  time,
            'nr_npa': nr_npa,
            'counter': counter,
            'ipos': np.reshape(np.fromfile(fid, 'float32', counter * 3),
                               (counter, 3), order='F'),
            'fpos': np.reshape(np.fromfile(fid, 'float32', counter * 3),
                               (counter, 3), order='F'),
            'v': np.reshape(np.fromfile(fid, 'float32', counter * 3),
                            (counter, 3), order='F'),
            'wght': np.fromfile(fid, 'float32', counter),
            'kind': np.fromfile(fid, 'float32', counter),
        }
    return data


def read_fbm(filename):
    """
    Read the Fast-ion distribution function file.

    Jose Rueda: jrrueda@us.es

    :param  filenam: name of the file to be read
    """
    print('Reading file: ', filename)
    with open(filename, 'rb') as fid:
        data = {
            'cdf_file': np.fromfile(fid, 'S17', 1)[:],
            'cdf_time': np.fromfile(fid, 'float64', 1)[:],
            'fbm_gc': int(np.fromfile(fid, 'int32', 1)[:]),
            'afbm': np.fromfile(fid, 'float64', 1)[:],
            # --- r grid
            'nr': int(np.fromfile(fid, 'int32', 1)[:]),
            'rmin': np.fromfile(fid, 'float64', 1)[:],
            'dr': np.fromfile(fid, 'float64', 1)[:],
        }
        data['cdf_file'] = data['cdf_file'][0].decode()
        data['rgrid'] = np.fromfile(fid, 'float64', data['nr'])[:]
        # --- z grid
        data['nz'] = int(np.fromfile(fid, 'int32', 1)[:])
        data['zmin'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['dz'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['zgrid'] = np.fromfile(fid, 'float64', data['nz'])[:]
        # --- fast-ion density
        data['denf'] = \
            np.reshape(np.fromfile(fid, 'float64', data['nr']*data['nz'])[:],
                       (data['nr'], data['nz']), order='F')
        # --- Energy grid
        data['nenergy'] = int(np.fromfile(fid, 'int32', 1)[:])
        data['emax'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['emin'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['dE'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['energy'] = np.fromfile(fid, 'float64', data['nenergy'])[:]
        # --- Pitch grid
        data['npitch'] = int(np.fromfile(fid, 'int32', 1)[:])
        data['pmax'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['pmin'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['dP'] = int(np.fromfile(fid, 'float64', 1)[:])
        data['pitch'] = np.fromfile(fid, 'float64', data['npitch'])[:]
        # --- 4D Fast-ion distribution
        data['fbm'] = \
            np.reshape(np.fromfile(fid, 'float32',
                                   data['nenergy'] * data['npitch']
                                   * data['nr'] * data['nz'])[:],
                       (data['nenergy'], data['npitch'],
                        data['nr'], data['nz']),
                       order='F')

    return data
