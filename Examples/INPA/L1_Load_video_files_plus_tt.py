"""
Load video from FILD cameras

Lesson 1 from the INPA experimental analysis. Video files will be loaded,
possibility to subtract noise and timetraces will be calculated

jose Rueda: jrrueda@us.es

Written for version 0.8.0

You should run paths_suite.py before runing this example.

Notice this is not machine independent as I will load signals from the AUG
database directly using the augshotfiles ignore last section and comment the
import aug_sfutils as sf in order to work with other machines
"""
import Lib as ss
import numpy as np
import matplotlib.pyplot as plt
import aug_sfutils as sf
import dd
# -----------------------------------------------------------------------------
# --- Section 0: Settings
# -----------------------------------------------------------------------------
# - General settings
shot = 40259
diag_ID = 1  # 6 for rFILD
t1 = 0.0     # Initial time to be loaded, [s]
t2 = 7.0     # Final time to be loaded [s]
limitation = True  # If true, the suite will not allow to load more than
limit = 2048       # 'limit' Mb of data. To avoid overloading the resources

# - Noise subtraction settings:
subtract_noise = True   # Flag to apply noise subtraction
tn1 = 0.0     # Initial time to average the frames for noise subtraction [s]
tn2 = 0.2     # Final time to average the frames for noise subtraction [s]

# - TimeTrace options:
t0 = 1.1          # time points to define the ROI
save_TT = False   # Export the TT and the ROI used

# - Plotting options:
plt_TT = True  # Plot the TT
# -----------------------------------------------------------------------------
# --- Section 1: Load video
# -----------------------------------------------------------------------------
# - open the video file:
vid = ss.vid.INPAVideo(shot=shot, diag_ID=diag_ID)
# - read the frames:
print('Reading camera frames: ')
vid.read_frame(t1=t1, t2=t2, limitation=limitation, limit=limit)

# -----------------------------------------------------------------------------
# --- Section 2: Subtract the noise
# -----------------------------------------------------------------------------
if subtract_noise:
    vid.subtract_noise(t1=tn1, t2=tn2)

# -----------------------------------------------------------------------------
# --- Section 3: Calculate the TT
# -----------------------------------------------------------------------------
# - Plot the frame
ax_ref = vid.plot_frame(t=t0)
fig_ref = plt.gcf()
# - Define roi
# Note: if you want the figure to re-appear after the selection of the roi,
# call create roi with the option re_display=True
roi = ss.tt.roipoly(fig_ref, ax_ref)
# Create the mask
mask = roi.getMask(vid.exp_dat['frames'][:, :, 0].squeeze())
# Calculate the TimeTrace
time_trace = ss.tt.TimeTrace(vid, mask)
# Save the timetraces and roi
if save_TT:
    print('Choose the name for the TT file (select .txt!!!): ')
    time_trace.export_to_ascii()
    print('Choose the name for the mask file: ')
    ss.io.save_mask(mask)

# -----------------------------------------------------------------------------
# --- Section 4: Plotting
# -----------------------------------------------------------------------------
if plt_TT:
    time_trace.plot_single(correct_baseline='ini')

# By default the time_trace.plotsingle() plot the sum of counts on the roi, but
# mean and std are also calculated and can be plotted with it, just explore a
# bit the function

# -----------------------------------------------------------------------------
# --- Section 5: Let's plot stuff with the shutter
# -----------------------------------------------------------------------------
# Get the opening shutter signals
NPI = sf.SFREAD(shot, 'NPI')
time = NPI('Time')
openShuter = np.array(NPI('ShutOpen')/16383.*20-10 < 1)
# Get the NBI3
NIS = sf.SFREAD(shot, 'NIS')
nbi3 = NIS('PNIQ')[:, 2, 0]
NBI3On = np.array(nbi3 < 1.5e6)
time3 = NIS.gettimebase('PNIQ')
fig, ax = plt.subplots(2, 1, sharex=True)

# --- Density and NBI3
DCN = dd.shotfile('DCN', shot)
h1 = DCN(b'H-1').data
h5 = DCN(b'H-5').data
timeh1 = DCN.getTimeBase('H-1')
ax[0].plot(timeh1, h1, label='H-1')
ax[0].plot(timeh1, h5, label='H-5')
ax[0].set_ylabel('Density')
ax[0].legend()
# --- Time Trace
time_trace.plot_single(ax=ax[1], correct_baseline='ini')
ax[1].fill_between(time3, 0, 1,
                   where=NBI3On,
                   alpha=0.25, color='r',
                   transform=ax[1].get_xaxis_transform())
ax[1].fill_between(time, 0, 1,
                   where=openShuter,
                   alpha=0.25, color='g',
                   transform=ax[1].get_xaxis_transform())
ax[1].set_xlim(t1, t2)
ax[1].set_ylim(0, time_trace.sum_of_roi.max()*1.05)

plt.show()

# --- Te and NBI3
IDA = dd.shotfile('IDA', shot)
te = IDA(b'Te').data
rho = IDA.getAreaBase(b'Te').data
irho = np.argmin(np.abs(rho[0, :] - 0.95))
print('Rho found: ', rho[0, irho])
timeTe = IDA.getTimeBase(b'Te')
fig1, ax1 = plt.subplots(2, 1, sharex=True)
ax1[0].plot(timeTe, te[:, 0], label='Te(0)')
ax1[0].plot(timeTe, te[:, irho], label='Te($\\rho=0.95$)')
ax1[0].set_ylabel('Temperature [eV]')
ax1[0].legend()
# --- Time Trace
time_trace.plot_single(ax=ax1[1], correct_baseline='ini')
ax1[1].fill_between(time3, 0, 1,
                    where=NBI3On,
                    alpha=0.25, color='r',
                    transform=ax1[1].get_xaxis_transform())
ax1[1].fill_between(time, 0, 1,
                    where=openShuter,
                    alpha=0.25, color='g',
                    transform=ax1[1].get_xaxis_transform())
ax1[1].set_xlim(t1, t2)
ax1[1].set_ylim(0, time_trace.sum_of_roi.max()*1.05)

plt.show()

# --- tau and NBI3
IDA = dd.shotfile('IDA', shot)
ne = IDA(b'ne').data
te = IDA(b'Te').data
rho = IDA.getAreaBase(b'Te').data
irho = np.argmin(np.abs(rho[0, :] - 0.95))
tau = 12.6e14*te**1.5/ne/16  # Assumed D and log Lambda_e=16
print('Rho found: ', rho[0, irho])
timeTe = IDA.getTimeBase(b'Te')
fig1, ax1 = plt.subplots(2, 1, sharex=True)
ax1[0].plot(timeTe, tau[:, 0], label='$\\tau(\\rho=0.00$)')
ax1[0].plot(timeTe, tau[:, irho], label='$\\tau(\\rho=0.95$)')
ax1[0].set_ylabel('Slowing down time [s]')
ax1[0].legend()
# --- Time Trace
time_trace.plot_single(ax=ax1[1], correct_baseline='ini')
ax1[1].fill_between(time3, 0, 1,
                    where=NBI3On,
                    alpha=0.25, color='r',
                    transform=ax1[1].get_xaxis_transform())
ax1[1].fill_between(time, 0, 1,
                    where=openShuter,
                    alpha=0.25, color='g',
                    transform=ax1[1].get_xaxis_transform())
ax1[1].set_xlim(t1, t2)
ax1[1].set_ylim(0, time_trace.sum_of_roi.max()*1.05)

plt.show()
