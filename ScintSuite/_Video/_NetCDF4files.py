"""
Routines to read pco (.b16) files.

Written by Jose Rueda: jrrueda@us.es
Adapted by Lina Velarde: lvelarde@us.es
"""
import os
import logging
import numpy as np
logger = logging.getLogger('ScintSuite.Video')
try:
    import netCDF4 as nc
except ModuleNotFoundError:
    print('netCDF4 library not found. Please install it.') 


def read_file_anddata(filename):
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
            print('No RFILD field in netcdf.')
            print('Will look in the logbook.')
        try:
            analoggain = data['gain'][:].data
        except IndexError:
            analoggain = 0
            print('No _gain_ field in netcdf.')
            try:
                print('Trying _analoggain_ field.')
                analoggain = data['analoggain'][:].data
                print('That worked! _analoggain_ field found.')
            except IndexError:
                print('No _analoggain_ field either. Setting to 0.')
                analoggain = 0
        try:
            print('Trying _diggain_ field.')
            digitalgain = data['diggain'][:].data
            print('That worked! _diggain_ field found.')
        except IndexError:
            print('No _diggain_ field. Setting to 0.')
            digitalgain = 0
    except: # it must be in standard format then
        vid = data['xfx']['video'][:].data
        time = data['xfx']['time'][:].data 
        timebase = (time[:]-time[0])/1e6 - 0.100 # in s
        fps = data['devices']['fps'][:].data
        exp = data['devices']['exposure'][:].data
        try:
            RFILD = data['devices']['RFILD'][:].data
        except IndexError:
            print('No RFILD field in netcdf.')
            print('Will look in the logbook.')
        try:
            analoggain = data['devices']['analoggain'][:].data
        except IndexError:
            print('No _analoggain_ field. Setting to 0.')
            analoggain = 0
        try:
            digitalgain = data['devices']['diggain'][:].data
        except IndexError:
            print('No _diggain_ field. Setting to 0.')
            digitalgain = 0
    
    data.close()
 
    frames = {'nf': vid.shape[2],
            'width': vid.shape[0], 
            'height': vid.shape[1], 
            'frames': vid,
            'timebase': timebase}

    imageheader = {
        'biWidth': vid.shape[0],
        'biHeight': vid.shape[1],
        'framesDtype': vid.dtype}
    header = {'ImageCount': vid.shape[2]}
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
            + ' the byte size of the variable!!!'
        print(text)
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
    print('Reading nc files')
    
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
        if video_object.file.endswith('.nc'): #not really need to check this, right?
            M[:, :, :] = load_nc(video_object.file)  
    else:
        # Load only the selected frames
        counter = 0
        if limitation and \
                size_frame * len(frames_number) > limit:
            print('Loading all frames is too much')
            return 0
        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      len(frames_number)),
                     dtype=video_object.imageheader['framesDtype'])

        # for j in frames_number:
        #     dummy = load_nc(video_object.file, j)
        #     M[:, :, counter] = dummy
        #     counter = counter + 1
        # print('Number of loaded frames: ', counter)

        # Would it be possible to do it like: ??
        dummy = load_nc(video_object.file, frames_number)
        M[:, :, :] = dummy
        print('Number of loaded frames: ', len(frames_number))
    return M
