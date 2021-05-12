"""Magnetic coils data"""
import Lib.LibData.AUG.DiagParam as params
import Lib.LibPaths as lpath
import dd                # Module to load shotfiles
import numpy as np
import scipy
from tqdm import tqdm

paths = lpath.Path()

def get_magnetics(shotnumber: int, coilNumber: int, coilGroup: str = 'B31',
                  timeWindow: float = None):
    """
    Retrieve from the shot file the magnetic data information.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: Shot number to get the data.
    @param coilNumber: Coil number in the coil array.
    @param coilGroup: can be B31, B17, C09,... by default set to B31
    (ballooning coils)
    @param timeWindow: Time window to get the magnetic data. If None, all the
    time window will be obtained.
    @return output: magnetic data (time traces and position.)
    """
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'

    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.

    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp,  edition=0)
    except:
        raise Exception('Shotfile not existent for '+diag+' #'+str(shotnumber))

    try:
        # Getting the time base.
        time = sf(name='Time')
    except:
        raise Exception('Time base not available in shotfile!')

    if timeWindow is None:
        timeWindow = [time[0], time[-1]]

    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])

    name = '%s-%02d' % (coilGroup, coilNumber)
    try:
        # Getting the time base.
        mhi = sf(name=name, tBegin=timeWindow[0], tEnd=timeWindow[1])
    except:
        raise Exception(name+' not available in shotfile.')
        sf.close()

    sf.close()
    # --- Getting the calibration factors from the CMH shotfile.
    # Get the last shotnumber where the calibration is written.
    cal_shot = dd.getLastShotNumber(diagnostic=b'CMH', pulseNumber=shotnumber,
                                    experiment=b'AUGD')

    sf = dd.shotfile(diagnostic='CMH', pulseNumber=cal_shot,
                     experiment='AUGD', edition=0)
    cal_name = 'C'+name
    cal = {
        'R': sf.getParameter(setName=cal_name, parName=b'R').data,
        'z': sf.getParameter(setName=cal_name, parName=b'z').data,
        'phi': sf.getParameter(setName=cal_name, parName=b'phi').data,
        'theta': sf.getParameter(setName=cal_name, parName=b'theta').data,
        'EffArea': sf.getParameter(setName=cal_name, parName=b'EffArea').data
    }

    t0 = np.abs(mhi.time-timeWindow[0]).argmin()
    t1 = np.abs(mhi.time-timeWindow[-1]).argmin()
    output = {
        'time': mhi.time[t0:t1],
        'data': mhi.data[t0:t1],
        'R': cal['R'],
        'z': cal['z'],
        'phi': cal['phi'],
        'theta': cal['theta'],
        'area': cal['EffArea']
    }
    
    # --- Ballooning coils phase correction
    # There is a phase correction to be applied for the B-coils.
    if (coilGroup == 'B31') and (coilNumber in params.mag_coils_phase_B31):
        A = np.loadtxt(paths.bcoils_phase_corr)
        
        idx = params.mag_coils_phase_B31.index(coilNumber)
        
        fv = A[2*idx, :]   # Vector frequency.
        ph = A[2*idx+1, :]  # Phase correction.
        fun = scipy.interpolate.interp1d(fv, ph, kind='linear', 
                                         fill_value=0.0, assume_sorted=True,
                                         bounds_error=False)
        
        output['phase_corr'] = { 'freq': fv,
                                 'phase': ph,
                                 'interp': fun
                               }

    sf.close()

    return output


def get_magnetic_poloidal_grp(shotnumber: int, timeWindow: float,
                              coilGrp: int = None):
    """
    It retrieves from the database the data of a set of magnetic coils lying
    within the same phi-angle. The way to choose them is either providing the
    coils group as provided above or giving the phi angle, and the nearest set
    of coils will be obtained. For example:
        phi_range = 45º will provide the C07 coil group.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param shotnumber: Shotnumber to retrieve the magnetic coils.
    @param timeWindow: time window to get the data.
    @param coilGrp: 'C07', 'C09', 'B31-5_11', 'B31-32-38' depending upon the
    group that needs to be read from database.
    @param phi_mag: each coil group is characterized by an approximate phi
    angle. If this is provided, the nearest group in phi is retrieved from the
    database. See variable @see{MHI_GROUP_APPR_PHI} to see which are the
    angles used.

    @return output: group of the magnetic coils.
    """
    # --- Parsing the inputs
    if coilGrp is None:
        raise Exception('Either the coil-group or the angle must be provided.')

    # If the time window is a single float, we will make an array containing
    # at least a 5ms window.
    dt = 5e-3
    if len(timeWindow) == 1:
        timeWindow = np.array((timeWindow, timeWindow + dt))

    # --- Parsing the magnetic group input
    # Check if it is in the list of coils.
    if coilGrp not in params.mag_coils_grp2coilName:
        raise Exception('Coil group, %s, not found!' % coilGrp)

    # Getting the list with the coils names.
    coil_list = params.mag_coils_grp2coilName[coilGrp]
    numcoils = len(coil_list[1])

    # --- Opening the shotfile.
    if shotnumber <= 33739:
        diag = 'MHA'
    else:
        diag = 'MHI'

    exp = 'AUGD'  # Only the AUGD data retrieve is supported with the dd.

    try:
        sf = dd.shotfile(diagnostic=diag, pulseNumber=shotnumber,
                         experiment=exp,  edition=0)
    except:
        raise Exception('Shotfile not existent for '+diag+' #'+str(shotnumber))

    try:
        # Getting the time base.
        time = sf(name='Time')
    except:
        raise Exception('Time base not available in shotfile!')

    if timeWindow is None:
        timeWindow = [time[0], time[-1]]

    timeWindow[0] = np.maximum(time[0], timeWindow[0])
    timeWindow[1] = np.minimum(time[-1], timeWindow[1])

    # --- Getting the calibration factors from the CMH shotfile.
    # Get the last shotnumber where the calibration is written.
    try:
        cal_shot = dd.getLastShotNumber(diagnostic=b'CMH',
                                        pulseNumber=shotnumber,
                                        experiment=b'AUGD')
        cal_sf = dd.shotfile(diagnostic='CMH', pulseNumber=cal_shot,
                             experiment='AUGD', edition=0)
    except:
        sf.close()
        raise Exception('Could not get the calibration data.')
        
        
        
    # Loading the b-coils corrections.
    if coilGrp[:3] == 'B31':
        A = np.loadtxt(paths.bcoils_phase_corr)
    # --- Getting the coils data.
    output = {
        'phi': np.zeros((numcoils,)),
        'theta': np.zeros((numcoils,)),
        'dtheta': np.zeros((numcoils,)),
        'R': np.zeros((numcoils,)),
        'z': np.zeros((numcoils,)),
        'area': np.zeros((numcoils,)),
        'time': [],
        'data': [],
        'coilNumber': np.zeros((numcoils,)),
        'phase_correction': list()
    }

    jj = 0
    flags = np.zeros((numcoils,), dtype=bool)
    for ii in tqdm(np.arange(numcoils)):
        name = '%s-%02d' % (coil_list[0], coil_list[1][ii])
        cal_name = 'C'+name

        try:
            # Try to get the magnetic data.
            mhi = sf(name=name, tBegin=timeWindow[0], tEnd=timeWindow[-1])

            # Try to get the calibration.
            cal = {
                'R': cal_sf.getParameter(setName=cal_name, parName=b'R').data,
                'z': cal_sf.getParameter(setName=cal_name, parName=b'z').data,
                'phi': cal_sf.getParameter(setName=cal_name,
                                           parName=b'phi').data,
                'theta': cal_sf.getParameter(setName=cal_name,
                                             parName=b'theta').data,
                'dtheta': cal_sf.getParameter(setName=cal_name,
                                              parName=b'dtheta').data,
                'area': cal_sf.getParameter(setName=cal_name,
                                            parName=b'EffArea').data
            }
        except:
            print('Could not retrieve coils %s-%02d' %
                  (coil_list[0], coil_list[1][ii]))
            continue

        t0 = np.abs(mhi.time-timeWindow[0]).argmin()   # Beginning time index.
        t1 = np.abs(mhi.time-timeWindow[-1]).argmin()  # Ending time index.

        if jj == 0:
            output['time'] = mhi.time[t0:t1]
            output['data'] = mhi.data[t0:t1]
        else:
            output['data'] = np.vstack((output['data'], mhi.data[t0:t1]))

        output['phi'][ii] = cal['phi']
        output['theta'][ii] = cal['theta']
        output['dtheta'][ii] = cal['dtheta']
        output['R'][ii] = cal['R']
        output['z'][ii] = cal['z']
        output['area'][ii] = cal['area']
        output['coilNumber'][ii] = ii+1
        flags[ii] = True
        
        # Appending the phase correction in case that it is needed.
        if (coilGrp[:3] == 'B31') and \
           (coil_list[1][ii] in params.mag_coils_phase_B31):
            idx = params.mag_coils_phase_B31.index(coil_list[1][ii])
            
            fv = A[2*idx, :]   # Vector frequency.
            ph = A[2*idx+1, :]  # Phase correction.
            flags = fv == np.nan
            fv = fv[~flags]
            ph = ph[~flags]
            fun = scipy.interpolate.interp1d(fv, ph, kind='linear',
                                             fill_value=0.0,
                                             assume_sorted=True,
                                             bounds_error=False)
            
            output['phase_corr'].append({ 'freq': fv,
                                          'phase': ph,
                                          'interp': fun
                                        })
        else:
            output['phase_corr'].append(dict())

        jj += 1

        del mhi
        del cal

    # --- All the coils have been read.
    # Removing the holes left by the coils that were not available in MHI.
    output['phi'] = output['phi'][flags]
    output['theta'] = output['theta'][flags]
    output['dtheta'] = output['dtheta'][flags]
    output['R'] = output['R'][flags]
    output['z'] = output['z'][flags]
    output['area'] = output['area'][flags]
    output['coilNumber'] = output['coilNumber'][flags]
    # --- Closing shotfiles.
    sf.close()
    cal_sf.close()

    return output
