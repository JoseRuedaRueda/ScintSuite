"""
Load video from FILD cameras


"""
import ScintSuite as ss
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 77971
diag_ID = 1  # There is only one fild at TCV

apd_channels = [1]

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.05     # Final time to average the frames for noise subtraction [s]

# - TimeTrace options:
t0 = 0.30         # time points to define the ROI
save_TT = False   # Export the TT and the ROI used

# - Plotting options:
FS = 16        # FontSize for plotting
plt_TT = True  # Plot the TT
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.FILDVideo(shot=shot, diag_ID=diag_ID, verbose=True)


# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
apd_channel = apd_channels[0]
path_file = 'Data/APD/TCV/APD1/' + '/ch_'+str(apd_channel)+'.txt'
path_in_pixels = ss.tt._roipoly.read_path_from_file(path_file = path_file)
roi = ss.tt.roipoly(path = path_in_pixels)
# # Create the mask
mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
######

import numpy as np

check = np.zeros((13,10))
apd_channels = range(129)

image = np.zeros((864,1280))
for i in range(128):
    apd_channel = apd_channels[i+1]
    print(apd_channel)
    path_file = 'Data/APD/TCV/APD1/' + '/ch_'+str(apd_channel)+'.txt'
    path_in_pixels = ss.tt._roipoly.read_path_from_file(path_file = path_file)
    roi = ss.tt.roipoly(path = path_in_pixels)
    # # Create the mask
    mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
    
    dummy = np.linspace(1,10,10)
    if (apd_channel) <117 and (apd_channel >13):
                    
        col = np.where((apd_channel-dummy*13)>=0);
        col = dummy[col[-1][-1]]
                  
        if (apd_channel-col*13)==0:
            row=13;    
            print('col is:', col)                     
        else:
            col = col +1;
            print('col is:', col) 
            row = apd_channel-(col-1)*13;
        
                        
    if apd_channel >=117:
        col = 10;
        if apd_channel-col*13==0:
            row = 13;
        else:
            row = apd_channel-9*13;
                   
    if apd_channel <=13:
        col = 1;
        if apd_channel-col*13==0:
            row = 13;
        else:
            row = apd_channel;
                    
    print('row is:', row)                            
    data_px = a[row-1,col-1]
    check[row-1,col-1] = data_px
    r, c = np.where(mask==True)
    image[r,c] = data_px
     
    
    

######
# # Calculate the TimeTrace
time_trace = ss.tt.TimeTrace(vid, mask)

# -- Save the timetraces and roi
if save_TT:
    print('Choose the name for the TT file (select .txt!!!): ')
    time_trace.export_to_ascii()
    print('Choose the name for the mask file: ')
    ss.io.save_mask(mask)

# -----------------------------------------------------------------------------
# --- Section 4: Plotting
# -----------------------------------------------------------------------------
if plt_TT:
    time_trace.plot_single()

# By default the time_trace.plotsingle() plot the sum of counts on the roi, but
# mean and std are also calculated and can be plotted with it, just explore a
# bit the function

# -----------------------------------------------------------------------------
# --- Extra
# -----------------------------------------------------------------------------
# Fourier analysis of the trace can be easily done:
# time_trace.calculate_fft()
# time_trace.plot_fft()
#
# time_trace.calculate_spectrogram()
# time_trace.plot_spectrogram()
plt.show()
