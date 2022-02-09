"""
Plot time traces of FILD fast-channels
"""
import Lib as ss
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# --- Settings
# -----------------------------------------------------------------------------

shot = [39610, 39611]  # figure with a subplot for each shot will pop-up

ch_to_plot = {  # Select the channels you want to plot
    1: [],
    2: [],
    4: [19],
    5: [],  # FILD 5 APD is broken
}
ptype = 'cloud'    # Type of plot: raw, smooth or cloud (this is raw + smooth)

filter = 'savgol'  # Type of filter, savgol, or median (bandpass coming soon)
points = 7500      # maximum number of points per channel to plot
# -----------------------------------------------------------------------------
# --- Load and plot channels
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(len(shot), sharex=True)
Signals = []
counter = 0  # To see in which shot we are
for sh in shot:   # Loop over shots
    for i in [1, 2, 4, 5]:   # Loop over FILDs
        if len(ch_to_plot[i]) > 0:
            dummy = ss.fc.FastChannel('FILD', i, ch_to_plot[i], sh)
            dummy.filter(filter)
            dummy.plot_channels(ptype=ptype, normalise=True, ax=ax[counter],
                                max_to_plot=points)

            Signals.append(dummy)
            del dummy
    ax[counter].set_ylim(0, 1)
    ax[counter].set_title('#' + str(shot[counter]))
    counter += 1

plt.show()
