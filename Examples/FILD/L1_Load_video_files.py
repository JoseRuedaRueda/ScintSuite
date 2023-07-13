"""
Load video from FILD cameras

Lesson 1 from the FILD experimental analysis. Video files will be loaded,
possibility to subtract noise

jose Rueda: jrrueda@us.es

Note; Written for version 0.1.8.  Revised for version 1.0.0
"""
import ScintSuite.as ss
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 44732
diag_ID = 1  # FILD manipulator number
t1 = 0.9     # Initial time to be loaded, [s]
t2 = 2.5     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.9     # Initial time to average the frames for noise subtraction [s]
tn2 = 1.0     # Final time to average the frames for noise subtraction [s]

# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID, verbose=True)
# Set the verbose to False if you do not want the console to print the
# comment written in the logbook by the FILD operator
# - read the frames:
print('Reading camera frames, shot: ', shot)
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)

# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    frame = vid.subtract_noise(t1=tn1, t2=tn2)
# The 'noise frame' used is returned as an output, but also, it is stored in
# the 'exp_dat' dictionary of the video object
# -----------------------------------------------------------------------------
# --- Extra
# -----------------------------------------------------------------------------
# There are 2 main plotting routines one could use at this point:
#       vid.GUI_frames()  # plot to see all the frames
#       vid.plot_frame(t=2.5) # Plot a single frame, accept custom color map,
#                             # Given ax to plot, etc...
#
# Of course there are much more, like plotting the number of saturated counts
# (usefult to see overheating) or other GUIS, just explore the different
# plotting routines (write 'vid.' and click tab to see all methods of the
# object)
