"""
This is a sample scrip which load a video and perform the remaping for each 
frame

Created as an example to use the routines without the graphical user interface

DISCLAIMER: This was created on the 11/11/2020. Since them several
improvements may have been done, it is possible that some function has been
changed and the script does not work at all now. If this happens, contact
jose rueda (jose.rueda@ipp.mpg.de) by email

"""

# General python packages
import numpy as np
import matplotlib.pyplot as plt
import LibPlotting as ssplt
import LibVideoFiles as ssvid
import LibMap as ssmap
import LibDataAUG as ssdat
import LibFILDSIM as ssFILDSIM
import sys
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib')
# from sf2equ_20200525 import EQU
# import mapeq_20200507 as meq
import map_equ as meq
# ------------------------------------------------------------------------------
# Section 0: Settings
# Paths:
cin_folder = '/p/IPP/AUG/rawfiles/FIT/'
calibration_database = './Calibrations/FILD/calibration_database.txt'
strike_map_folder = '/afs/ipp-garching.mpg.de/home/r/ruejo/FILD_Strike_maps/'
# calibration data
camera = 'PHANTOM'
cal_type = 'PIX'
diag_ID = 1     # FILD Number
# Shot a time interval
t1 = 0.5        # Initial time to remap [in s]
t2 = 2.5        # Final time to remap [in s]
shot = 32326
# remapping parameters
gmin = 1.2      # Minimum gyroradius [in cm]
gmax = 10.5     # Maximum gyroradius [in cm]
deltag = 0.1    # Interval of the gyroradius [in cm]

pmin = 20.0     # Minimum pitch angle [in degrees]
pmax = 90.0     # Maximum pitch angle [in degrees]
deltap = 1.0    # Pitch angle interval
# Position of the FILD
rfild = 2.0
zfild = 0.30
alpha = 0.0
beta = -12.0



# ------------------------------------------------------------------------------

# %% Section 1: Load calibration
database = ssmap.CalibrationDatabase(calibration_database)
cal = database.get_calibration(shot, camera, cal_type, diag_ID)
# ------------------------------------------------------------------------------

# %% Section 2: Load video file and the necesary frames
# Prepare the name of the cin file to be loaded
dummy = str(shot)
file = cin_folder + dummy[0:2] + '/' + dummy +'_v710.cin'
cin = ssvid.Video(file)
it1 = np.array([np.argmin(abs(cin.timebase-t1))])
it2 = np.array([np.argmin(abs(cin.timebase-t2))])
frames = cin.read_frame(np.arange(start=it1, stop=it2+1, step=1))
frame_shape = frames[:, :, 1].shape
nframes = frames.shape[2]
# If the last line fails because you tried to load just one time point... you
# are using the wrong Example file... plase read the name
# ------------------------------------------------------------------------------

# %% Section 3: remap the shot

# Initialise the matix to save the results. I allways get confused with the
# fact that an array ini:end:delta x has end-ini/delta or end-ini/delta+1
# elements... It is late and I am hungry, this is just and example so YOLO, I
# will create the arrays and take: length
gyrdum = np.arange(start=gmin, stop=gmax, step=deltag)
pitdum = np.arange(start=pmin, stop=pmax, step=deltap)

ngyr = len(gyrdum) - 1
npit = len(pitdum) - 1

# open the magnetic field shot file
equ = meq.equ_map(shot, diag='EQH')


for iframe in range(nframes):
    # Load the magnetic field
    tframe = cin.timebase[it1 + iframe]
    br, bz, bt, bp = ssdat.get_mag_field(shot, rfild, zfild, time=tframe,
                                         equ=equ)
    phi, theta = ssFILDSIM.calculate_fild_orientation(br, bz, bt, alpha, beta)
    print(phi,theta)
# %%
    

# smap = ssmap.StrikeMap(0, strike_map)
# smap.calculate_pixel_coordinates(cal)
# smap.interp_grid(frame_shape, plot=False, method=1)
# # ------------------------------------------------------------------------------

# # %% Section 3: test of the remapping algorithm
# # Load a frame
# cin = ssvid.Video(cin_file_name)
# t0 = 2.50
# dummy = np.array([np.argmin(abs(cin.timebase-t0))])
# ref_frame = cin.read_frame(dummy)
# # Perform the remaping with the default options
# remaped, pitch, energy = ssmap.remap(smap, ref_frame)
# # Plot the remapped frame
# plt.contourf(pitch, energy, remaped.T)
