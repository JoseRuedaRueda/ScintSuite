"""
Routines to read XIMEA (.nc) files.

Written by Jose Rueda: jrrueda@us.es
Adapted by Lina Velarde: lvelarde@us.es
"""
import os
import pyuda
import logging
import numpy as np
logger = logging.getLogger('ScintSuite.Video')
try:
    import netCDF4 as nc
except ModuleNotFoundError:
    print('netCDF4 library not found. Please install it.') 

client = pyuda.Client()



def read_file_anddata(connection = None, filename = None):
    """
    Read netCDF4 files from MU format. 
    It will try to read the old format (before shot ~48580), and if that fails will
    read the standard format.
    Read info for a case where the measurements are stored as .nc
    Return a series of dictionaries, with all the info we can extract from the XIMEA files.

    Lina Velarde - lvelarde@us.es

    :param filename: path+filename of the camera data
 
    @return image_header: dictionary containing the info about the image size,
    and shape
    @return header: Header 'similar' to the case of a cin file
    @return settings: dictionary similar to the case of the cin file (it only
    contains the exposition time)
    Outputs:
        timebase array in s
        fps
        exposure in us
    """
    RFILD = None
    beta_angle = None
    if filename is not None:
        data = nc.Dataset(filename, 'r')
        try: #try reading old format
            vid = data['video'][:].data
            time = data['time'][:].data
            timebase = (time[:]-time[0])/1e6 - 0.100 # in s
            fps = data['fps'][:].data
            exp = data['exposure'][:].data
            try:
                RFILD = data['RFILD'][:].data
            except IndexError:
                logger.warning('No RFILD field in netcdf.')
                logger.warning('Will look in the logbook.')
            try:
                analoggain = data['gain'][:].data
            except IndexError:
                analoggain = 0
                logger.warning('No _gain_ field in netcdf.')
                try:
                    analoggain = data['analoggain'][:].data
                except IndexError:
                    analoggain = 0
            try:
                digitalgain = data['diggain'][:].data
            except IndexError:
                digitalgain = 0
            logger.info('Reading old format of the nc file.')
        except: # it must be in standard format then
            logger.info('Reading standard format of the nc file.')
            vid = data['xfx']['video'][:].data
            time = data['xfx']['time'][:].data 
            # in the standard format, there have been two different ways to write
            # the timebase. The first one was in ms and then converted to s, 
            # by correcting it to 0 and removing the trigger time (0.1 s), while the
            # second one was written directly in s and the corrections were made before
            # writing it to the file. We need to check which type we are reading here.
            # If in ms, all elements are positive. If in s, the first element is negative 
            # (due to the trigger).
            if np.all(time[:]>0): 
                timebase = (time[:]-time[0])/1e6 - 0.100 # in s
                # print log message that this is the old format
                logger.info('Reading old format of the XIMEA timebase. This will be deprecated.')
            elif time[0] < 0: # if the first element is negative, it is already in s
                timebase = time[:]
                logger.info('Reading new format of the XIMEA timebase.')
            else:
                raise Exception('Timebase is not in the expected format. Please check the file.')        
            fps = data['devices']['fps'][:].data
            exp = data['devices']['exposure'][:].data
            try:
                RFILD = data['devices']['RFILD'][:].data
            except IndexError:
                logger.warning('No RFILD info in netcdf. Will look in the logbook.')
            try:
                beta_angle = data['devices']['FILDangle'][:].data
            except IndexError:
                logger.warning('No beta angle info in netcdf. Will look in the logbook.')

            try:
                analoggain = data['devices']['analoggain'][:].data
            except IndexError:
                logger.warning('No _analoggain_ field. Setting to 0.')
                analoggain = 0
            try:
                digitalgain = data['devices']['diggain'][:].data
            except IndexError:
                logger.warning('No _diggain_ field. Setting to 0.')
                digitalgain = 0
        
        data.close()
 
    else:
        logger.info('Reading standard format of the nc file from UDA.')
        vid = connection['xfx']['video'].data
        n0 = vid.shape[0]
        n1 = vid.shape[1]
        n2 = vid.shape[2]
        vid = vid.flatten()
        vid = np.reshape(vid, (n2, n1, n0))
        time = connection['xfx']['time'].data 
        # in the standard format, there have been two different ways to write
        # the timebase. The first one was in ms and using the internal camera clock,
        # and then converted to s, by shifting it to 0 and removing the trigger time (0.1 s), 
        # while the second one was written directly in s as the corrections were made before
        # writing it to the file. We need to check which type we are reading here.
        # If in ms, I think all elements are either positive or negative. 
        # If in s, the first element is negative (due to the trigger), the last one
        # is postive, and it should span around 1-10 seconds, more would mean it's not correct.
        
        # -- Basic sanity checks before writing time
        min_raw = time.min()
        max_raw = time.max()
        range_raw = max_raw - min_raw
        # 1) Check if time is always positive -> ms
        if np.all(time > 0):
            # Normal case: convert counts to seconds
            timebase = (time - time[0]) / 1e6 - 0.100
            logger.info('Reading old format of the XIMEA timebase (all positive counts).')
        # 2) If the first value is negative, it will be probably corrected. 
        # But make sure by checking the total range
        # time is negative or suspicious, try to detect if the scale is wrong:
        elif min_raw < 0:
            # Check if range_raw is tiny (e.g. < 10 seconds) or huge (indicates not corrected)
            if range_raw < 10:
                timebase = time[:]
                logger.info('Reading new format of the XIMEA timebase.')
            else:
                # Large range means it's suspicious, probably ms => raise warning
                timebase = (time - time[0]) / 1e6 - 0.100
                logger.warning(f"Timebase range is suspiciously large ({range_raw}), \
                    correcting to seconds. But it'd be better to check!")
        else:
            # Something else unexpected
            raise ValueError("Timebase is not in the expected format. Please check the file.")
        # 3) Additional sanity check: timebase values should be monotonic and increasing
        if not np.all(np.diff(timebase) >= 0):
            logger.warning("Timebase is not monotonically increasing. This could cause errors.")
        # 4) Additional sanity check: timebase should be within reasonable bounds (e.g., max < 1000 s)
        if timebase.max() > 1000 or timebase.min() < -1000:
            logger.warning("Timebase max is suspiciously large (> 1000 s). Check units and data.")
    
        fps = connection['devices']['fps'].data
        exp = connection['devices']['exposure'].data
        try:
            RFILD = connection['devices']['RFILD'].data
        except KeyError:
            logger.warning('No RFILD info in netcdf. Will look in the logbook.')
        try:
            beta_angle = connection['devices']['FILDangle'].data
        except KeyError:
            logger.warning('No beta angle info in netcdf. Will look in the logbook.')

        try:
            analoggain = connection['devices']['analoggain'].data
        except KeyError:
            logger.warning('No _analoggain_ field. Setting to 0.')
            analoggain = 0
        try:
            digitalgain = connection['devices']['diggain'].data
        except KeyError:
            logger.warning('No _diggain_ field. Setting to 0.')
            digitalgain = 0

    frames = {'nf': vid.shape[2],
            'width': vid.shape[0], 
            'height': vid.shape[1], 
            'frames': vid,
            'timebase': timebase}

    imageheader = {
        'biWidth': vid.shape[0],
        'biHeight': vid.shape[1],
        'framesDtype': vid.dtype}
    if RFILD is None and beta_angle is None:
        header = {'ImageCount': vid.shape[2]}
    elif RFILD is not None and beta_angle is None:
        header = {
        'ImageCount': vid.shape[2],
        'R_FILD': np.round(float(RFILD),4)}
    elif beta_angle is not None and RFILD is None:
        header = {
        'ImageCount': vid.shape[2],
        'beta_angle': np.round(beta_angle,4)}
    else:
        header = {
        'ImageCount': vid.shape[2],
        'R_FILD': np.round(float(RFILD),4),
        'beta_angle': np.round(beta_angle,4)}

    # Possible bytes per pixels for the camera
    BPP = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'int32': 32, 'uint64': 64}
    settings = {
            'fps': fps,
            'exp': exp,
            'digitalgain': digitalgain,
            'analoggain': analoggain}
    try:
        settings['RealBPP'] = BPP[imageheader['framesDtype'].name]
        text = 'In the nc there is no info about the real BitesPerPixel'\
            + ' used in the camera. Assumed that the BPP coincides with'\
            + ' the byte size of the variable.'
        logger.warning(text)
    except KeyError:
        raise Exception('Expected uint8,16,32,64 in the frames')
 
    return frames, header, imageheader, settings


def load_nc(filename: str, frame_number: int = None):
    """
    Load the nc with an order compatible with IDL 
    It will try to read the old format (before shot ~48580), and if that fails will
    read the standard format.

    IDL load things internally in a way different from python. In order the new
    suite to be compatible with all FILD calibrations of the last 15 years,
    an inversion should be done to load png in the same way as IDL

    @param filename: full path pointing to the nc
    """
    data = nc.Dataset(filename, mode='r')
    if frame_number is None:
        try:
            dummy = data['video'][:]
        except:
            dummy = data['xfx']['video'][:]
    else:
        try:
            dummy = data['video'][..., frame_number]
        except:
            dummy = data['xfx']['video'][..., frame_number]

    return dummy[::-1, ...]


def read_frame(video_object, frames_number=None, limitation: bool = True,
               limit: int = 2048, verbose: bool = True):
    """
    Read .nc files

    Jose Rueda: jrrueda@us.es
    Lina Velarde: lvelarde@us.es

    @param video_object: Video class with the info of the video. See the video
        object of the BasicVideoObject.py file
    @param frames_number: array with the number of the frames to be loaded,
    if none, all frames will be loaded
    @param limitation: if we want to set a limitation of the size we can load
    @param limit: Limit to the size, in megabytes

    @return M: array of frames, [px in x, px in y, number of frames]
    """
    # Frames would have a name as CCD_qe_frame.b16 example: CCD_qe_0001.b16
    logger.info('Reading nc files')
    
    # check the size of the files, data will be saved as float32
    size_frame = video_object.imageheader['biWidth'] * \
        video_object.imageheader['biWidth'] * 2 / 1024 / 1024
    if frames_number is None:
        # In this case, we load everything
        if limitation and \
                size_frame * video_object.header['ImageCount'] > limit:
            raise Exception('Loading all frames is too much')
            return 0

        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      video_object.header['ImageCount']),
                     dtype=video_object.imageheader['framesDtype'])
        M[:, :, :] = load_nc(video_object.file)  
    else:
        # Load only the selected frames
        counter = 0
        if limitation and \
                size_frame * len(frames_number) > limit:
            logger.warning('Loading all frames is too much')
            return 0
        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      len(frames_number)),
                     dtype=video_object.imageheader['framesDtype'])

        dummy = load_nc(video_object.file, frames_number)
        M[:, :, :] = dummy
        logger.info(f'Number of loaded frames: {len(frames_number)}')
    return M
