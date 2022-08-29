"""
Routines to read .png files.

Written by Jose Rueda: jrrueda@us.es

These routines are just a wrapper for standard python methods just to leave the
data in the same order (colums and rows) from the old IDL FILD analysis
routines, in order to preserve the database variables
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import Lib._Video._AuxFunctions as aux
from skimage import io                     # To load images


def read_data(path):
    """
    Read info for a case where the measurements are stored as png

    Jose Rueda Rueda: jrrueda@us.es

    Return a series of dictionaries similar to the case of a cin file,
    with all the info we can extract from the png

    @param path: path to the folder where the pngs are located
    @return time_base: time base of the frames (s)
    @return image_header: dictionary containing the info about the image size,
    and shape
    @return header: Header 'similar' to the case of a cin file
    @return settings: dictionary similar to the case of the cin file (it only
    contain the exposition time)
    """
    # Look for a png to extract the size and a .txt for the time information
    f = []
    look_for_png = True
    for file in os.listdir(path):
        if file.endswith('.txt'):
            f.append(os.path.join(path, file))
        if file.endswith('.png') and look_for_png:
            dummy = io.imread(os.path.join(path, file))
            si = dummy.shape
            imageheader = {
                'biWidth': si[0],
                'biHeight': si[1],
                'framesDtype': dummy.dtype}
            look_for_png = False
    # If no png was found, raise and error
    if look_for_png:
        print('No .png files in the folder...')
        return 0, 0, 0, 0
    n_files = len(f)
    if n_files == 0:
        print('no txt file with the information found in the directory!!')
        return 0, 0, 0, 0
    elif n_files > 1:
        print('Several txt files found...')
        return 0, 0, 0, 0
    else:
        dummy = np.loadtxt(f[0], skiprows=2, comments='(')
        header = {'ImageCount': int(dummy[-1, 0])}
        # @ToDo, this should be /1e9, right?
        settings = {'ShutterNs': dummy[0, 2] / 1000.0}
        # Possible bytes per pixels for the camera
        BPP = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}
        try:
            settings['RealBPP'] = BPP[imageheader['framesDtype'].name]
            text = 'In the PNG there is no info about the real BitesPerPixel'\
                + ' used in the camera. Assumed that the BPP coincides with'\
                + ' the byte size of the variable!!!'
            print(text)
        except KeyError:
            raise Exception('Expected uint8,16,32,64 in the frames')
        time_base = dummy[:, 3]
        # --- Check for problemns in the timebase:
        # - P1: Time base completelly broken:
        # Sometimes the camera break and set the says that all the frames
        # where recorded at t = 0... in this case just assume a time base on
        # the basis of the exposition time.
        # Usually in this situation the camera records fine, it just its time
        # module which is broken
        std_time = np.std(time_base)
        # Test if all the exposure time was the same
        if std_time < 1e-2 and np.mean(time_base) < 0.1:
            time_base = np.linspace(0, dummy[-1, 0] * dummy[0, 2] / 1000,
                                    int(dummy[-1, 0]))
            print('Caution!! the experimental time base was broken, a time '
                  'base has been generated on the basis of the exposure time')
        # - P2: Time base partially broken:
        # some times after recording some frames, the camera fail and record
        # 'broken frames'. Usually also the time points for these broken frames
        # is zero (example 34597 at 4.48s). To ensure we are not in one of
        # these cases, call the check time base function:
        if aux.check_timebase(time_base):
            print('The time base seems broken!!!!')
            print('Time is not always increasing!')
            print('What do you want to do?: ')
            print('1: Plot the time base')
            print('Otherwhise: Continue without plotting')
            p = input('Enter the answer: ')
            if int(p) == 1:
                fig, ax = plt.subplots()
                ax.plot(time_base, label='Original')
                ax.set_ylabel('Time [s]')
                ax.set_xlabel('Frame number')
                fig.show()
                # note, spyder is bugged, so the figure will not be shown until
                # the end of the execution, therefore, I include here this
                # ginput with a limit of 1s, this will force the window to
                # appear and the user will not notice this 1 second stop :-)
                plt.ginput(timeout=1)

            print('Now what?: ')
            print('0: Ignore those frames')
            print('1: Include a fake time base for those frames')
            print('Otherwise: Continue with this weird time base')
            a = int(input('Enter the answer: '))
            if a == 0 or a == 1:  # Find the first point where this was broken
                dif = np.diff(time_base)
                flags = dif < 0
                id = np.arange(len(time_base), dtype=int)
                id = id[1:]
                limit = id[flags]
                limit = int(limit[:])
            if a == 0:  # Ignore the frames if needed
                header['ImageCount'] = limit
                time_base = time_base[:limit]
            if a == 1:  # Change the timebase
                tb = time_base.copy()
                time_base = np.linspace(0, dummy[-1, 0] * dummy[0, 2] / 1000,
                                        int(dummy[-1, 0]))
                time_base[:limit] = tb[:limit]
            # Plot the new timebase
            if int(p) == 1:
                ax.plot(time_base, label='Considered')
                fig.show()
                plt.ginput(timeout=1)
    return header, imageheader, settings, time_base[:].flatten()


def load_png(filename: str):
    """
    Load the png with an order compatible with IDL

    IDL load things internally in a way different from python. In order the new
    suite to be compatible with all FILD calibrations of the last 15 years,
    an inversion should be done to load png in the same way as IDL

    @param filename: full path pointing to the png
    """
    dummy = io.imread(filename)
    if len(dummy.shape) > 2:     # We have an rgb png, transform it to gray
        dummy = aux.rgb2gray(dummy)

    return dummy[::-1, :]


def read_frame(video_object, frames_number=None, limitation: bool = True,
               limit: int = 2048):
    """
    Read .png files

    Jose Rueda: jrrueda@us.es

    @param video_object: Video class with the info of the video.  See the video
        object of the BasicVideoObject.py file
    @param frames_number: array with the number of the frames to be loaded,
    if none, all frames will be loaded
    @param limitation: if we want to set a limitation of the size we can load
    @param limit: Limit to the size, in megabytes

    @return M: array of frames, [px in x, px in y, number of frames]
    """
    # Frames would have a name as shot-framenumber.png example: 30585-001.png
    print('Reading PNG files')
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
        counter = 0
        for file in sorted(os.listdir(video_object.path)):
            if file.endswith('.png'):
                M[:, :, counter] = load_png(
                    os.path.join(video_object.path, file))
                counter = counter + 1
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
            if file.endswith('.png'):
                current_frame = current_frame + 1
                if current_frame in frames_number:
                    pngname = os.path.join(video_object.path, file)
                    dummy = load_png(pngname)
                    M[:, :, counter] = dummy
                    counter = counter + 1
                if counter == video_object.header['ImageCount']:
                    break
        print('Number of loaded frames: ', counter)
    return M
