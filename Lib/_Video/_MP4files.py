"""
Routines to read mp4 files with ffmpeg.

Written by Hannah Lindl: hannah.lindl@ipp.mpg.de

"""
import numpy as np
import subprocess as sp


def read_file(video, filename_video: str):
    """
    load greyscale camera data with ffmpeg
    @param video: video properties containing the camera resolution and the timebase 
    @param filename_video: path/filename of the camera data 
    """

    time = video.timecal
    width = video.properties['width']
    height = video.properties['height']
    nf = video.nf
    
    initial_time = 0


    FFMPEG_BIN = 'ffmpeg'
    command = [FFMPEG_BIN,
                 '-loglevel', 'error',
                 '-hide_banner',
                '-ss', str(initial_time),
               '-i', filename_video,
                '-frames:v', str(nf),
               '-f', 'image2pipe',
                '-pix_fmt', 'gray16le',
               '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command,stdout = sp.PIPE, bufsize = 10**9)
    raw_image = pipe.stdout.read(nf*2*width*height)
    image=np.frombuffer(raw_image, np.uint16).reshape([-1, height, width])
    pipe.stdout.flush()


    frames = (image.astype(float))

    return {'nf': video.nf, 'nx': width, 'ny': height, 'frames': frames, 'tframes': time}
