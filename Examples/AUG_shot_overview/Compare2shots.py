"""
Basic comparison of 2 shots

Jose Rueda: jrrueda@us.es

Done for the IAEA presentation
"""
import dd
import matplotlib.pyplot as plt

shots = [32312, 39612]
tmax = 2.0  # maximum time to plot
# The 5th plot is for the user to plot the FILD timetrace
fig, ax = plt.subplots(5, sharex=True)


for shot in shots:
    # --- Axis 1: Plasma current
    Ip_shotfile = dd.shotfile('MAG', shot)
    Ip = Ip_shotfile(b'Ipa')
    Vloop = Ip_shotfile(b'ULid12')
    Ip_shotfile.close()
    ax[0].plot(Ip.time, Ip.data/1.0e6, label='Ip [MA]')
    ax[0].set_ylim(0, 1.0)
    # ax[0].set_ylabel('Ip [MA]')
    # --- Axis 2: Edge density
    DCN = dd.shotfile('DCN', shot)
    H5 = DCN(b'H-5')
    DCN.close()
    ax[1].plot(H5.time, H5.data/1.0e19)
    ax[1].set_ylim(0, 3.0)
    # ax[1].set_ylabel('$\\bar n_e$ [1e19 $m^{-2}$]')
    # --- Axis 3: NBI power
    NIS = dd.shotfile('NIS', shot)
    PNI = NIS(b'PNI')
    NIS.close()
    ax[2].plot(PNI.time, PNI.data/1.e6)
    ax[2].set_ylim(0, 3.0)
    # ax[2].set_ylabel('$P_{NBI}$ [MW]')
    # --- Axis 4: Vloop
    ax[3].plot(Vloop.time, Vloop.data)
    ax[3].set_ylim(0, 15.0)
    # ax[3].set_ylabel('$V_{loop}$ [V]')
    ax[4].set_xlabel('Time [s]')
    # --- Axis 5: Future FILD timetrace
    # To be added by the user
    ax[4].set_xlim(0, tmax)

# remove the space between plots
plt.subplots_adjust(hspace=0)
# show figure
fig.show()
