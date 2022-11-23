"""
Routines to read mp4 files with ffmpeg.

Written by Hannah Lindl: hannah.lindl@ipp.mpg.de

"""
import numpy as np
import subprocess as sp
import ffmpeg

def read_file(filename_video: str):
    """
    Load greyscale camera data with ffmpeg.

    Hannah Lindl - hannah.lindl@ipp.mpg.de

    :param filename_video: path/filename of the camera data
    """

    FFMPEG_BIN = 'ffmpeg'
    command = [FFMPEG_BIN,
                 '-loglevel', 'error',
                 '-hide_banner',
                 '-i', filename_video,
                 '-f', 'image2pipe',
                 '-pix_fmt', 'gray16le',
                 '-vcodec', 'rawvideo', '-']

    width = int(ffmpeg.probe(filename_video)["streams"][0]['width'])
    height = int(ffmpeg.probe(filename_video)["streams"][0]['height'])
    fps = int(ffmpeg.probe(filename_video)["streams"][0]['avg_frame_rate'][:-2])
    nf = int(ffmpeg.probe(filename_video)['streams'][0]['nb_frames'])


    pipe = sp.Popen(command,stdout = sp.PIPE, bufsize = 10**9)
    raw_image = pipe.stdout.read(nf*2*width*height)
    image=np.frombuffer(raw_image, np.uint16).reshape([-1, height, width])
    pipe.stdout.flush()


    frames = (image.astype(float))

    return {'nf': nf, 'width': width, 'height': height, 'frames': frames, 'fps': fps}
