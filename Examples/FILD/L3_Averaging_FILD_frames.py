"""
Average FIlD frames to reduce noise

Lesson 3 of the FILD data-analysis. Video frames will be averaged. Ideally for
situations with lot of noise

Created for version 0.8.0
Revised for version 1.0.0
"""
import Lib as ss

# --- Settings
# - General settings
shot = 39612
diag_ID = 1  # 6 for rFILD
t1 = 0.2     # Initial time to be loaded, [s]
t2 = 1.0     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.20     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.23     # Final time to average the frames for noise subtraction [s]
flag_copy = False  # If true, a copy of the frames will be done while
#                  # substracting the noise

# - Filter options:
apply_filter = True  # Flag to apply filter to the frames
kind_of_filter = 'median'
options_filter = {
    'size': 1        # Size of the window to apply the filter
}

# --- Load the video
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID)
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)
# --- Subtract the noise
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)
# --- Prepare the window to average:
# Option 1: uniform window, for example of 50 ms
window = vid.generate_average_window(0.05)
# Option2: You can give your trace as inputs and select your arbitrary windows
# with a giput. See vid.generate_average_window? to see how th trace should be
# stored
# window = vid.generate_average_window(trace=mytrace)
# Now average the frames
vid.average_frames(window)

# --- Launch the GUI to see the averaged frames
vid.GUI_frames(flagAverage=True)  # Notice the flagAverage, not present in L1
