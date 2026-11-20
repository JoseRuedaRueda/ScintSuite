"""
Routines to read mp4 files with ffmpeg.

Written by Hannah Lindl: hannah.lindl@ipp.mpg.de

"""
import logging
import numpy as np
logger = logging.getLogger("ScintSuite.Video.MP4files")
try:
    import ffmpeg
except ImportError:
    logger.warning("0: FFmpeg is not installed. Please install it to use the MP4 module.")
import os
try:
    from skvideo.io import vread as video_read
except ImportError:
    logger.warning("0: scikit-video is not installed. Please install it to use the MP4 module.")
try:
    import av
except ImportError:
    logger.warning("0: av is not installed. Please install it to use the MP4 module.")

def read_file(fn: str, force_gray: bool=True, bpp: int=None, pix_fmt: str=None,
              bits_size: int=None, **kwargs):
    """
    Wrap to use the reading rutines

    Alex Reyner: areyner@us.es
    """

    functions = (('PyAV', read_file_pyav), ('ffmpeg', read_file_ffmpeg),)

    last_error = None
    for name, reader in functions:
        try:
            return reader(fn=fn, force_gray=force_gray, bpp=bpp, 
                          pix_fmt=pix_fmt, bits_size=bits_size, **kwargs,)
        except Exception as err:
            last_error = err
            logger.warning(f'{name} failed: {err}')

    raise RuntimeError(f'All mp4 read_file failed for {fn}') from last_error

def read_file_pyav(fn: str, force_gray: bool=True, bpp: int=None, pix_fmt: str=None,
              bits_size: int=None, **kwargs):
    """
    Load the data with av
    Issue, truncates to 8bit data

    Alex Reyner: areynerv@us.es

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

    container = av.open(fn)
    try:
        stream = container.streams.video[0]
        fps_raw = float(stream.average_rate) if stream.average_rate else 0.0
        frame_count = int(stream.frames) if stream.frames else 0

        frames_list = []

        for packet in container.demux(stream):
            for frame in packet.decode():
                # Keep full bit depth when converting pixel format
                target_fmt = 'gray16le' if force_gray else 'rgb48le'
                arr = frame.to_ndarray(format=target_fmt)

                # If original sample depth fits in 8 bits, keep uint8; otherwise uint16
                src_bits = frame.format.components[0].bits if frame.format.components else 8
                if src_bits <= 8:
                    arr = (arr >> 8).astype(np.uint8)
                else:
                    # gray16le / rgb48le places high-bit samples in MSB-aligned 16-bit words
                    shift = max(0, 16 - src_bits)
                    arr = (arr >> shift).astype(np.uint16)

                frames_list.append(arr.squeeze())
    finally:
        container.close()

    if not frames_list:
        raise ValueError(f'No frames could be read from: {fn}')

    video = np.stack(frames_list, axis=0)

    nf = len(video)
    height = video.shape[1]
    width = video.shape[2]

    if (bpp is None) and (pix_fmt is None) and (bits_size is None):
        if video.dtype == np.uint8:
            out_dtype = np.float16
        elif video.dtype == np.uint16:
            out_dtype = np.float32
        elif video.dtype == np.uint32:
            out_dtype = np.float64
        elif np.issubdtype(video.dtype, np.floating):
            out_dtype = video.dtype
        else:
            out_dtype = float

        frames = video.astype(dtype=out_dtype)
    else:
        if bits_size is None:
            bits_size = video.dtype.itemsize * 8

        target_dtype = {8: np.uint8, 16: np.uint16}.get(bits_size, video.dtype)
        arr = video.astype(target_dtype, copy=False)

        shift = 0 if bpp is None else max(0, bits_size - bpp)
        if shift > 0:
            arr = np.right_shift(arr, shift)

        frames = arr.squeeze()

    return {
        'nf': nf if frame_count <= 0 else min(nf, frame_count),
        'width': width,
        'height': height,
        'frames': frames,
        'fps': int(round(fps_raw)) if fps_raw > 0 else 0,
        'colored': frames.ndim > 3,
        'transformed_to_gray': force_gray,
        'bits_per_pixel': bpp,
    }

def read_file_ffmpeg(fn: str, force_gray: bool=True, bpp: int=None, pix_fmt: str=None,
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
        video = np.right_shift(np.frombuffer(out, dtype).reshape(shape), shift)

        # dtype = 'float%d' % (2*int(np.ceil((bits_size - bpp)/8)))

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
