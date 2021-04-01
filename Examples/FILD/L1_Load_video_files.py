"""
Load video from FILD cameras

Lesson 1 from the FILD experimental analysis. Video files will be loaded,
possibility to substract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.1.8
"""
import Lib as ss
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 38686
diag_ID = 6  # 6 for rFILD (DLIF)
t1 = 0.9     # Initial time to be loaded, [s]
t2 = 8.0     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise substraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.9     # Initial time to average the frames for noise subtraction [s]
tn2 = 1.0     # Final time to average the frames for noise subtraction [s]

# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - Get the proper file name
filename = ss.vid.guess_filename(shot, ss.dat.FILD[diag_ID-1]['path'],
                                 ss.dat.FILD[diag_ID-1]['extension'])

# - open the video file:
vid = ss.vid.Video(filename, diag_ID=diag_ID)
# - read the frames:
print('Reading camera frames: ')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)

# -----------------------------------------------------------------------------
# --- Section 2: Substract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Extra
# -----------------------------------------------------------------------------
# There are 2 main plotting routines one could use at this point:
#       vid.GUI_frames()  # plot to see all the frames
#       vid.plot_frame(t=2.5) # Plot a single frame, accept custom color map,
#                             # Given ax to plot, etc...
