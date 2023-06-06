"""
Routines to read mp4 files with ffmpeg.

Written by Hannah Lindl: hannah.lindl@ipp.mpg.de

"""
import numpy as np
import ffmpeg
import os
from skvideo.io import vread as video_read

def read_file(fn: str, force_gray: bool=True, bpp: int=None, pix_fmt: str=None,
              bits_size: int=None, **kwargs):
    """
    Load the data from the video using the ffmpeg package.

    Balazs Tal - balazs.tal@ipp.mpg.de
    Hannah Lindl - hannah.lindl@ipp.mpg.de
    Pablo Oyola - poyola@us.es

    :param fn: path/filename of the camera data.
    :param force_gray: if the data is originally RGB, this will transform it
    into a gray scale.
    :param bpp: bits per pixel. If None, the full size of the video is considered
    as useful data.
    :param pix_fmt: pixel format. If None, the pixel format of the video is
    considered.
    :param bits_size: bits per pixel to actually read from the file. If None,
    the full size of the video is considered as useful data.
    :param kwargs: stub.
    """

    prop = ffmpeg.probe(fn)['streams'][0]
    width = int(prop['width'])
    height = int(prop['height'])
    fps = int(prop['avg_frame_rate'].split('/')[0].strip())
    nf  = int(prop['nb_frames'])

    print(f'Dynamic range = {prop["bits_per_raw_sample"]} bits')
    print(f'Pixel format = {prop["pix_fmt"]}')

    if (bpp is None) and (pix_fmt is None) and (bits_size is None):
        # We rely on scikit video reader.
        video = video_read(fn, as_grey=force_gray).squeeze()

        # Getting the minimum floating point type to store the data.
        if video.dtype == np.uint8:
            dtype = np.float16
        elif video.dtype == np.uint16:
            dtype = np.float32
        elif video.dtype == np.uint32:
            dtype = np.float64
        elif video.dtype == np.float32:
            dtype = np.float32
        elif video.dtype == np.float64:
            dtype = np.float64
        else:
            dtype = float

        # Saving the video as a numpy array in float.
        frames = video.astype(dtype=dtype)
    else:
        if pix_fmt is None:
            pix_fmt = prop['pix_fmt']
        if bits_size is None:
            bits_size = int(prop['bits_per_raw_sample'])

        dtype = { 8: np.uint8,
                  16: np.uint16
                }.get(bits_size)

        shape = [nf, height, width, -1]

        out = ffmpeg.input(fn).output('pipe:', format='rawvideo',
                                    pix_fmt=pix_fmt).run(quiet=True)[0]

        if bpp is None:
            shift = 0
        else:
            shift = bits_size - bpp
        video = np.right_shift(np.frombuffer(out, bits_size).reshape(shape), shift)

        dtype = 'float%d' % (2*int(np.ceil((bits_size - bpp)/8)))

        frames = video.astype(dtype=dtype).squeeze()

    output = { 'nf': nf, #  Number of frames.
               'width': width, # Number of pixels along horizontal.
               'height': height, # Number of pixels along the vertical.
               'frames': frames, # Frame data.
               'fps': fps, # Frames per second.
               'colored': frames.ndim > 3, # Whether the image is RGB.
               'transformed_to_gray': force_gray,
               'bits_per_pixel': bpp,
             }

    return output

def write_file(fn: str, video: float, bit_size: int=16, bpp: int=None,
               encoding: str=None, fps: int=120):
    """
    Writes to file a given buffer provided the properties of the video.

    Pablo Oyola - poyola@us.es

    :param fn: output filename. An error is raised if the file exists before
    creation.
    :param video: a 3-dim array with (time, pix_x, pix_y).
    :param dtype: type to write to the video.
    :param bpp: bits per pixel to actually write to file.
    :param encoding: how to write the file. If this is None, two situations
    appear
        - If the input video is 4D, then RGB encoding is used.
        - If the input video is 3D, gray little endian is used.
    """

    if os.path.isfile(fn):
        raise FileExistsError('File %s already exists!'%fn)

    # Checking the type of video.
    if video.ndim < 3:
        raise ValueError('The video must have dimension at least 3.')
    elif video.ndim == 3:
        color = False
    elif video.ndim == 4:
        color = True
    else:
        raise ValueError('The video must have as much, size 4')

    if encoding is None:
        if color:
            pix_fmt = 'rgb%d'%bit_size + 'le'
        else:
            pix_fmt = 'gray%d'%bit_size + 'le'

    else:
        if color and (not encoding.lower().beginswith('rgb')):
            raise ValueError('The input frame is colored but the pixel ' + \
                             'encoding is gray-scale')

        pix_fmt = encoding.lower()+'%d'%bit_size + 'le'

    # Checking dtype.
    dtype = {8: np.uint8,
             16: np.uint16
             }.get(bit_size)

    # Transforming the output to write.
    data = video.astype(dtype)
    if color:
        ntime, width, height, _ = data.shape
    else:
        ntime, width, height = data.shape

    size = f'{width}x{height}'

    # Creating the pipe to write:
    proc = (
             ffmpeg
             .input('pipe: ', format='rawvideo', pix_fmt=pix_fmt, s=size)
             .output(fn, pix_fmt=pix_fmt)
             .overwrite_output()
             .run_async(pipe_stdin=True)
           )

    proc.stdin.write(data.tobytes())
    proc.stdin.close()
    proc.wait()


