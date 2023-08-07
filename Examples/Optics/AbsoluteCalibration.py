"""
Perform the absolute calibration using measurements of the integrating sphere

It is written for the case of a video taken with the phantom camera, some minor
modification may be needed in section 1 if you use another video format

Jose Rueda: jrrueda@us.es

"""

import numpy as np
import ScintSuite.as ss
import scipy.interpolate as interp


# -----------------------------------------------------------------------------
# --- Settings and paths
# -----------------------------------------------------------------------------
video_path = '/afs/ipp/home/r/ruejo/INPA_calibration/data/'\
    + 'Videos_phantom_camera/esfera_a_34mm_strikemap_unpacked.cine'

sphere_path = '/afs/ipp/home/r/ruejo/ScintSuite/Data/'\
    + 'IntegratingSpheres/Sphere_AUG_30cm_6.5A_04_09_2019.dat'

camera_response_path = '/afs/ipp/home/r/ruejo/ScintSuite/Data/'\
    + 'CamerasSpectralResponse/PhantomV2512.txt'

c_to_e = 5   # Number of electrons needed to have a count
threshold_to_identify = 1000  # threshold [counts] to identify the sphere
area_sphere = np.pi * 0.0504 ** 2  # [m**2]
# -----------------------------------------------------------------------------
# --- Section 1: Load the video
# -----------------------------------------------------------------------------
# - Open video and load frames
vid = ss.vid.Video(video_path, shot=0)
vid.read_frame(np.arange(vid.header['ImageCount']))
# - cut the video to select just the sphere region
vid.cut_frames(420, 670, 460, 690, make_copy=False)
# - Prepare the mask of the sphere
frame = \
    ss.vid.binary_image(vid.exp_dat['frames'][:, :, 0], threshold_to_identify)
# - calculate the time trace of that region
trace = ss.tt.TimeTrace(vid)
trace_current = trace.sum_of_roi * c_to_e * ss.par.ec\
    / vid.settings['ShutterNs'] * 1e9

# -----------------------------------------------------------------------------
# --- Section 2: sphere and camera data
# -----------------------------------------------------------------------------
cam = ss.optics.read_spectral_response(camera_response_path, True)
sph = ss.optics.read_sphere_data(sphere_path, True)
# Interpolate camera data in the same Wl axis:
inter1D = interp.interp1d(cam['lambda'], cam['response'], fill_value=0,
                          bounds_error=False)
cam_response = inter1D(sph['lambda'])
delta_l = sph['lambda'][1] - sph['lambda'][0]
ideal_current = np.sum(cam_response * sph['spectrum'] * delta_l * area_sphere)

# -----------------------------------------------------------------------------
# --- calculate the transmission
# -----------------------------------------------------------------------------
t = trace_current / ideal_current
print('Your transmission factor is: ', t.mean(), '[sr^-1]')
