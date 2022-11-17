"""
Routines to read .tif files.

Written by Jose Rueda: jrrueda@us.es

These routines are just a wrapper for standard python methods just to leave the
data in the same order (colums and rows) from the old IDL FILD analysis
routines, in order to preserve the database variables

Under development
"""
import os
import f90nml
import logging
import numpy as np
import Lib._Video._AuxFunctions as aux
from skimage import io                     # To load images

# --- Auxiliary objects
logger = logging.getLogger('ScintSuite.Video')

def read_data(path):
    """To Be implemented."""
    f = []
    look_for_tiff = True
    for file in os.listdir(path):
        if file.endswith('.txt'):
            f.append(os.path.join(path, file))
        if file.endswith('.tif') and look_for_tiff:
            dummy = io.imread(os.path.join(path, file))
            si = dummy.shape
            imageheader = {
                'biWidth': si[0],
                'biHeight': si[1],
                'framesDtype': dummy.dtype}
            look_for_tiff = False
    # If no tif was found, raise and error
    if look_for_tiff:
        print('No .tif files in the folder...')
        return 0, 0, 0, 0
    n_files = len(f)
    if n_files == 0:
        print('no txt file with the information found in the directory!!')
        return 0, 0, 0, 0
    elif n_files > 1:
        print('Several txt files found, loop to find the namelist...')
        print('if there are 2 fortran namelist, this can give unwanted result')
        for file in f:
            name = os.path.join(path, file)
            print('Trying with: %s' % name)
            nml = f90nml.read(name)
            if len(nml.keys()) == 0:
                continue   # This is not the file we were looking for
            print('Success!')
            nml = nml['config']  # just take the main namelist dict
    else:  # Just one txt, assume is the good one
        name = os.path.join(path, f[0])
        nml = f90nml.read(name)
        nml = nml['config']

    time_base = np.arange(nml['nframes']) * nml['exposure_time'] \
        + nml['trigger']
    header = {'ImageCount': nml['nframes']}
    settings = {'ShutterNs': nml['exposure_time'] / 1000.0}
    # Possible bytes per pixels for the camera
    BPP = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}
    try:
        settings['RealBPP'] = BPP[imageheader['framesDtype'].name]
        text = 'In the Tiff there is no info about the real '\
            + 'BitesPerPixel used in the camera. Assumed that the BPP'\
            + ' coincides with the byte size of the variable!!!'
        print(text)
    except KeyError:
        print(imageheader['framesDtype'].name)
        raise Exception('Expected uint8,16,32,64 in the frames')
    return header, imageheader, settings, time_base[:].flatten()


def load_tiff(filename: str):
    """
    Load the tiff files

    Assume there is only one frame per file

    :param  filename: full path pointing to the tiff

    :return frame: loaded frame
    """
    dummy = io.imread(filename)
    if len(dummy.shape) > 2:     # We have an rgb tiff, transform it to gray
        dummy = aux.rgb2gray(dummy)

    return dummy[::-1, :]

def load_tiff_singleFile(filename: str):
    """
    Load the tiff files

    Assume all frames are stored in the same file, and we have b/w camera

    :param  filename: full path pointing to the tiff

    :return frame: loaded frame
    """
    dummy = io.imread(filename)

    return np.moveaxis(dummy[:, ::-1, :], 0, 2)


def read_frame(video_object, frames_number=None, limitation: bool = True,
               limit: int = 2048):
    """
    Read .tiff files

    Jose Rueda: jrrueda@us.es

    :param  video_object: Video class with the info of the video.  See the video
        object of the BasicVideoObject.py file
    :param  frames_number: array with the number of the frames to be loaded,
    if none, all frames will be loaded
    :param  limitation: if we want to set a limitation of the size we can load
    :param  limit: Limit to the size, in megabytes

    :return M: array of frames, [px in x, px in y, number of frames]
    """
    # Frames would have a name as shot-framenumber.png example: 30585-001.png
    logger.info('Reading TIF files')
    # check the size of the files, data will be saved as float32
    size_frame = video_object.imageheader['biWidth'] * \
        video_object.imageheader['biWidth'] * 2 / 1024 / 1024
    # Count how many tiff files we have in the folder
    totalCounter = 0
    for file in sorted(os.listdir(video_object.path)):
        if file.endswith('.tif'):
            totalCounter += 1
    if totalCounter == 0:
        raise Exception('My dear friend, there are no tiff files')
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
        # Count how many tiff files we have in the folder
        counter = 0
        for file in sorted(os.listdir(video_object.path)):
            if file.endswith('.tif'):
                if totalCounter > 1:
                    M[:, :, counter] = load_tiff(
                        os.path.join(video_object.path, file))
                    counter = counter + 1
                else:
                    M = load_tiff_singleFile(
                            os.path.join(video_object.path, file))
                    break
            if counter == video_object.header['ImageCount']:
                break
    else:
        # Load only the selected frames
        counter = 0
        current_frame = 0
        if limitation and \
                size_frame * len(frames_number) > limit:
            print('Loading all frames is too much')
            return 0
        M = np.zeros((video_object.imageheader['biWidth'],
                      video_object.imageheader['biHeight'],
                      len(frames_number)),
                     dtype=video_object.imageheader['framesDtype'])

        for file in sorted(os.listdir(video_object.path)):
            if file.endswith('.tif'):
                current_frame = current_frame + 1
                if current_frame in frames_number:
                    pngname = os.path.join(video_object.path, file)
                    dummy = load_tiff(pngname)
                    M[:, :, counter] = dummy
                    counter = counter + 1
                if counter == video_object.header['ImageCount']:
                    break
        print('Number of loaded frames: ', counter)
    return M
