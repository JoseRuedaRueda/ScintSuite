"""
Routines to read mp4 files.

Written by Jose Rueda: jrrueda@us.es

Just a small rapper to cv2 capabilities
"""
import numpy as np
import Lib._Video._AuxFunctions as aux

try:
    import cv2
except ImportError:
    print('There would be no support for the mp4 videos, open cv not found')


def read_file(file, verbose: bool = True):
    """
    Read frames and time base from an mp4 file

    Jose Rueda: jrrueda@us.es

    @param file: full path to the file
    @param verbose: flag to print some info in the terminal

    @return: Dictionary containing the frames of the video:
        - 'nf': Number of loaded frames
        - 'nx': Height of the frame
        - 'ny': Width of the frame
        - 'frames': Frames [nx, ny, nf]
        - 'tframes': Time of each frame
    """
    # --- Open the video file
    vid = cv2.VideoCapture(file)

    # --- Get the number of frames in the video
    nf = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # --- Get the frame rate
    fr = vid.get(cv2.CAP_PROP_FPS)
    if verbose:
        print('We will load: ', nf, ' frames')
        print('Frame rate: ', fr)
    # --- Read the frames
    nx = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ny = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

    time = np.zeros(nf)
    frames = np.zeros((nx, ny, nf))
    counter = 0
    success = True
    while success:
        success, image = vid.read()
        if success:
            time[counter] = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # if frame is in rgb, transform to gray
            if len(image.shape) > 1:
                image = aux.rgb2gray(image)
            frames[:, :, counter] = image
            counter += 1
    return {'nf': nf, 'nx': nx, 'ny': ny, 'frames': frames, 'tframes': time}
