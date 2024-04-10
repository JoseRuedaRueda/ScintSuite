"""
Other routines.

This library will contain routines that because of its nature do not belong to
a particular category.
This library contains:

- FILD4 trajectories routines.
- ELM starting points.
- ICRH power

"""

import numpy as np
import xarray as xr

import ScintSuite.errors as errors

import ScintSuite._Video._MATfiles as mat
try:
    import MDSplus as mds
    import sys
    sys.path.append('/home/jansen/NoTivoli/ascot-tcv/tcvpy/')
    import tcv
except:
    pass   
from ScintSuite._Paths import Path
import ScintSuite.errors as errors
import xarray as xr
from datetime import datetime

pa = Path()

# -----------------------------------------------------------------------------
# --- RETRIEVING VIDEO DATA FROM TCV MAT FILES
# -----------------------------------------------------------------------------
def read_MAT_video_data(file: str):
    '''
    Until 2024 TCV FILD video data are stored as .mat files
    Two structures exist, one for the XIMEA camera operated on pcfild002 (until end 2023)
        and on pcfild004 (since start of2024)
        and one for the APD camera operated on pcfild003
    
    Important: Need to be on LAC9 to acces videodata directory
    '''
    mat_out = xr.Dataset()
    if ('pcfild002' in file) or ('pcfild004' in file):
        dummy = mat.read_file(file)
        t0 = dummy['/b/secs'][0][1] - dummy['/b/secs'][0][0] + (dummy['/b/usecs'][0][1] - dummy['/b/usecs'][0][0])*1e-6
        timebase = t0 + dummy['/b/secs'][0] - dummy['/b/secs'][0][0] + (dummy['/b/usecs'][0] - dummy['/b/usecs'][0][0])*1e-6

        frames = dummy.pop('/dat') #frames are stored in "/dat"
        frames = frames[:,:,::-1]  #flip the image in the y direction.
        
        nt, nx, ny = frames.shape

        # Matplotlib imshow considers the first index to be the rox index. So we relable to axis to fit this convention
        py = np.arange(nx)
        px = np.arange(ny)
        mat_out['frames'] = \
            xr.DataArray(np.transpose(frames, axes = [2, 1, 0]), dims=('px', 'py', 't'),
                        coords={'t': timebase.squeeze(),
                                'px': px,
                                'py': py})
        mat_out['nframes'] = xr.DataArray(np.arange(nt), dims=('t'))
        mat_out['RealBPP'] = 10

        del dummy
        del frames
    else:
        '''
        load APD data
        '''
        data = get_APD(file)
        timebase = data['time']

        #The APD has 8 by 16 pixels, however the fibre bundle viewing the scintillator is 10 by 13.
        nt, nx, ny = len(timebase), 10, 13   #The scintillator 
        apd_data = np.array(data['data'])
        #It's more intuative to work in the scintilator matrix, therefore we add two dummy channels since we need to go from 128 to 130
        apd_data = np.append(apd_data.flatten(), np.zeros( nt*2 ) )
        #apd_data = np.reshape(apd_data, (nt) )
        apd_data = np.reshape(apd_data, (nx, ny, nt) )

        px = np.arange(nx)
        py = np.arange(ny)
        mat_out['frames'] = \
            xr.DataArray(apd_data, dims=('px', 'py', 't'),
                        coords={'t': timebase.squeeze(),
                                'px': px,
                                'py': py})
        mat_out['nframes'] = xr.DataArray(np.arange(nt), dims=('t'))
        mat_out['RealBPP'] = 10

        del data
        del apd_data

    return mat_out

# -----------------------------------------------------------------------------
# --- GENERIC SIGNAL RETRIEVING.
# -----------------------------------------------------------------------------
def get_signal_generic(shot: int,
                       signame: str = 'None', 
                       edition: int = 0,
                       tBegin: float = None,
                       tEnd: float = None,
                       **kwargs):
    """
    Function that generically retrieves a signal from the database in TCV.

    Pablo Oyola - pablo.oyola@ipp.mpg.de
    Anton Jansen van Vuuren - anton.jansenvanvuuren@epfl.ch

    :param  shot: shotnumber of the shotfile to read.
    :param  diag: not needed
    :param  signame: MDS path 
    :param  exp: experiment where the shotfile is stored. Default to AUGD.
    :param  edition: edition of the shotfile to open. If 0, the last closed
    edition is opened.
    :param  tBegin: initial time point to read.
    :param  tEnd: final time point to read.
    """

    # Reading the second diagnostic data.

    #MDS_tdi_command = r'pd_calibrated(1)'
    #tree = mds.Tree('tcv_shot', shot)
    #data = tree.getNode(dataPath).data()
    #time = tree.getNode(dataPath).dim_of().data()

    if signame == None:
        MDS_tdi_command = r'pd_calibrated(1)'
    else:
        MDS_tdi_command = signame

    MDS_Connection = tcv.shot(shotnum = shot)
    data = MDS_Connection.tdi(MDS_tdi_command).values
    time = MDS_Connection.tdi('dim_of(%s)'%MDS_tdi_command).values
    MDS_Connection.close()
    #if not sfo.status:
    #    raise errors.DatabaseError('The signal data cannot be read for #%05d:%s:%s(%d)'
    #                    % (shot, diag, signame, edition))
    
    if data is None:
        raise errors.DatabaseError('Cannot find signal %s' % signame)

    if tBegin is None:
        t0 = 0
    else:
        t0 = np.abs(time - tBegin).argmin()

    if tBegin is None:
        t1 = len(time)
    else:
        t1 = np.abs(time - tEnd).argmin()

    data = np.array(data[t0:t1, ...], dtype=float)
    time = np.array(time[t0:t1, ...], dtype=float)

    return time, data


# -----------------------------------------------------------------------------
# --- SIGNAL OF FAST CHANNELS.
# -----------------------------------------------------------------------------
def get_fast_channel(diag: str, diag_number: int, channels, shot: int,
                     ed: int = 0, exp: str = 'AUGD'):
    """
    Get the signal for the fast channels (APD)

    Anton Jansen van Vuuren: anton.jansenvanvuuren@epfl.ch

    :param  channels: channel number we want, or arry with channels
    :param  shot: shot file to be opened
    """

    # Look which channels we need to load:
    try:    # If we received a numpy array, all is fine
        nch_to_load = channels.size
        if nch_to_load == 1:
            # To solve the bug that just one channel is passed but as a
            # component of a numpy array
            ch = np.array([channels]).flatten()
        else:
            ch = channels
    except AttributeError:  # If not, we need to create it
        ch = np.array([channels]).flatten()
        nch_to_load = ch.size

    # Open the shot file

    file = '/videodata/pcfild003/data/APD%i.mat'%shot
    dummy = mat.read_file(file)
    time = np.linspace(0, dummy['/b/meas_length'][0][0], dummy['/b/n_data'][0][0])

    data = dummy.pop('/dat') 
    data = -data[:, ch].T  ##APD data is negative so multiply with -1

    return {'time': time, 'data': data, 'channels': ch}


def get_APD(file: str):
    """
    Get the signal for the fast channels (APD)

    Anton Jansen van Vuuren: anton.jansenvanvuuren@epfl.ch

    :param  shot: shot file to be opened
    """

    # Open the shot file

    dummy = mat.read_file(file)
    time = np.linspace(0, dummy['/b/meas_length'][0][0], dummy['/b/n_data'][0][0])

    data = dummy.pop('/dat') 
    data = 2**14-data[:, :].T  ##APD data is negative so multiply with -1


    ##APD channels are not organized in MAT file. We have to use a mapping matrix to reorganise
    mapping_matrix = [
        [91, 20, 17,18, 56, 55, 127, 53, 59, 116, 113, 114, 24, 23, 95, 21],
        [19, 92, 89, 90, 128, 126, 54, 125, 115, 60, 57, 58, 96, 94, 22, 93],
        [14, 69, 13, 15, 41, 42, 43, 100, 110, 37, 109, 111, 9, 10, 11, 68],
        [70, 16, 72, 71, 97, 98, 99, 44, 38, 112, 40, 39, 65, 66, 67, 12],
        [76, 3, 2, 1, 103, 104, 48, 102, 108, 35, 34, 33, 7, 8, 80, 6],
        [4, 75, 74, 73, 47, 45, 101, 46, 36, 107, 106, 105, 79, 77, 5, 78],
        [29, 86, 30, 32, 122, 121, 124, 51, 61, 118, 62, 64, 26, 25, 28, 83],
        [85, 31, 87, 88, 50, 49, 52, 123, 117, 63, 119, 120, 82, 81, 84, 27]
        ]

    mapping_matrix = np.array(mapping_matrix)
    data_mapped =data[mapping_matrix.flatten() - 1, :]

    return {'time': time, 'data': data_mapped, 'channels': np.arange(128)}


# -----------------------------------------------------------------------------
# --- ELMs
# -----------------------------------------------------------------------------
def get_ELM_timebase(shot: int, time: float = None, **kwargs):
    """
    Give the ELM onset and duration times
    Use DEFUSE matlab package from Alessandro Pau
    Anton Jansen van Vuuren 

    :param  shot: shot number
    :returns tELM: Dictionary with:
        -# t_onset: The time when each ELM starts
        -# dt: the duration of each ELM
        -# n: The number of ELMs
    """
    # --- Open the AUG shotfile
    '''
    First run the ELM detection matlab script of Alessandro
    (https://gitlab.epfl.ch/spc/defuse/-/tree/feature/DEFUSE?ref_type=heads)
    cd NoTivoli/jansen
    matlab

    shot = XXX
    setup_DEFUSE_paths
    setup_DEFUSE_paths;
    [table_tcv,SXR]=get_sig_data_TCV({shot});

    save('/NoTivoli/jansen/SF/ELM/XXX.mat', 'ELM')
    '''
    file = '/NoTivoli/jansen/SF/ELM/%i.mat'%shot

    try:
        ELM = mat.read_file(file)
        #ELM['ELM'][0][0][0]
    except:
        raise Exception('Cannot access shotfile %s:#%05d:ELM' % (exp, shot))
        
    t_onset =  ELM['ELM'][0][0][0][0]
    t_peak =  ELM['ELM'][0][0][1][0]
    t_end =  ELM['ELM'][0][0][2][0]
    tELM = {
        't_onset': t_onset,
        'dt': np.diff(t_onset),
        'energy': np.zeros(len(t_onset)),
        'f_ELM': 1/np.diff(t_onset)
    }

    if time is not None:
        # If one time is given, we find the nearest ELM.
        time = np.atleast_1d(time)
        if len(time) == 1:
            t0 = np.abs(tELM['t_onset'] - time[0]).argmin()
            tELM = {
                't_onset': np.array((tELM['t_onset'][t0],)),
                'dt': np.array((tELM['dt'][t0],)),
                'energy': np.array((tELM['energy'][t0],)),
                'f_ELM': np.array((tELM['f_ELM'][t0],))
            }
        elif len(time) == 2:
            t0, t1 = np.searchsorted(tELM['t_onset'], time)
            t1 = min(len(tELM['t_onset']), t1+1)
            tELM = {
                't_onset': tELM['t_onset'][t0:t1],
                'dt': tELM['dt'][t0:t1],
                'energy': tELM['energy'][t0:t1],
                'f_ELM': tELM['f_ELM'][t0:t1]
            }

        else:
            tidx = [np.abs(tELM['t_onset'] - time_val).argmin()
                    for time_val in time]

            tELM = {
                't_onset': tELM['t_onset'][tidx],
                'dt': tELM['dt'][tidx],
                'energy': tELM['energy'][tidx],
                'f_ELM': tELM['f_ELM'][tidx]
            }

    tELM['n'] = len(tELM['t_onset'])

    return tELM

# -----------------------------------------------------------------------------
# --- OTHER USEFUL SIGNALS.
# -----------------------------------------------------------------------------
def get_neutron(shot: int, time: float = None, exp: str = 'AUGD', 
                xArrayOutput: bool = False):
    """"
    Reads the neutron rates for a given AUG shot in a given time interval.

    Pablo Oyola - poyola@us.es

    @param shot: shotnumber to get the neutron rate.
    @param time: time where to retrieve the neutron rate. 
        - If None, all the time points are return
        - If a single time point is provided, then it is interpolated to
         that point
        - If two time points are given, then the signal is returned in that 
        time range.
        - If a time basis is provided, then the windowed average is used.
    """

    if time is None:
        tBegin = None
        tEnd   = None
        time_mode = 0
    else:
        time = np.atleast_1d(time)
        if len(time) == 1:
            tBegin = time - 1.0e-2
            tEnd   = time + 1.0e-2
            time_mode = 1

        elif len(time) == 2:
            tBegin, tEnd = time.tolist()
            time_mode = 2
        else:
            tBegin, tEnd = time.min(), time.max()
            time_mode = 3
    
    # Reading the data from the shotfile.
    time_out, data = get_signal_generic(shot=shot, diag='ENR', 
                                        signame='NRATE_II', exp=exp,
                                        tBegin=tBegin, tEnd=tEnd)
    
    if time_mode == 1:
        data = np.interp(time, time_out, data)
        time_out = time
    elif time_mode == 2:
        bins = len(time)
        ranges = [tBegin, tEnd]
        data = np.histogram(time_out, bins=bins, range=ranges,
                            weights=data)
        norm = np.histogram(time_out, bins=bins, range=ranges)

        data /= norm
        time_out = np.linspace(tBegin, tEnd, bins)

    # We also compute the total neutron emitted during the whole time range
    if time_mode != 1:
        tot = np.trapz(data, time_out)
    else:
        tot = None

    # Preparing the output.
    if xArrayOutput:
        time = xr.DataArray(time_out, attrs={'units': 's'},
                            name='Time')

        attrs = { 'units': 'neutron/s',
                  'diag': 'ENR',
                  'exp': exp,
                  'total': tot
                }
        output = xr.DataArray(data, coords=(time,), 
                              dims=(time.name,), 
                              attrs=attrs, name='Neutron rate')
    else:
        output = { 'time': time,
                   'data': data,
                   'units': 'neutron/s',
                   'diag': 'ENR',
                   'exp': exp,
                   'base_units': 's',
                   'total': tot
                 }

    return output
    

def get_neutron_history(shot_0: int, shot_1: int):
    """
    Compute for all the shot in the range of shots given, the evolution of the
    neutron emitted.

    Pablo Oyola - poyola@us.es

    @param shot_0: starting shot.
    @param shot_1: ending shot to return.
    """

    if shot_0 == shot_1:
        shots = (shot_0,)
    else:
        shots = np.arange(shot_0, shot_1+1)
    
    shots = np.array(shots, dtype=int)
    nshots = len(shots)

    # Calling systematically the database to get the total 
    # neutron rate.
    neutron_evol = np.zeros_like(shots)
    abstime = np.zeros_like(shots, dtype=object)
    flags = np.ones_like(shots, dtype=bool)
    for ii, ishot in enumerate(shots):
        try:
            tmp = get_neutron(ishot, xArrayOutput=True)
            neutron_evol[ii] = tmp.total
        except errors.DatabaseError:
            neutron_evol[ii] = 0.0
            flags[ii] = False
        
        jou = sf.journal.getEntryForShot(int(ishot))
        if 'upddate' not in jou:
            abstime[ii] = np.nan
            flags[ii] = False
        else:
            abstime[ii] = datetime.strptime(jou['upddate'].decode(), 
                                            '%Y-%m-%d %H:%M:%S\n')

    # Removing useless points
    abstime = abstime[flags]
    neutron_evol = neutron_evol[flags]
    time = np.zeros_like(neutron_evol, dtype=float)

    # Preparing the output:
    for ii in range(len(neutron_evol)):
        time[ii] = (abstime[ii] - abstime[0]).total_seconds()

    sortidx = np.argsort(time)
    time = time[sortidx]
    neutron_evol = neutron_evol[sortidx]

    return time, neutron_evol
    
