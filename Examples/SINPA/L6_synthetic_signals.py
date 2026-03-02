"""
Example to calculate synthetic signals with the FMC class

Alex Reyner

Distortion and other advanced optic effects are not included yet
"""

import os
import numpy as np
import ScintSuite as ss
import matplotlib.pyplot as plt
import matplotlib
import ScintSuite._Plotting as ssplt

import ScintSuite.SimulationCodes.SINPA._ForwardModellingClass as fmod
import ScintSuite.SimulationCodes.SINPA._forward_modelling as fmod2

matplotlib.use('QtAgg')
plt.close('all')
plt.ion()

# ----------------------------------------------------------------------------
# %% Satrt object
# ----------------------------------------------------------------------------
# First do a FILDSIM simulation to obtain an strikemap.
# Load smap
input = 'ITERFILD_mk9_c_i'
smap_input = ss.smap.Fsmap(file=os.path.join(ss.paths.SINPA,'runs',input,'results','%s.map'%input))
# Load smap to plot, if wanted
toplot = 'ITERFILD_mk9_c_i_s'
smapplt_input = ss.smap.Fsmap(file=os.path.join(ss.paths.SINPA,'runs',toplot,'results','%s.map'%toplot))
# Load scintillator, give geometry path
scintillator = ss.scint.Scintillator(file=os.path.join(ss.paths.SINPA,'Geometry/ITERFILD_mk0/Element1.txt'), format='sinpa', particle='Alpha')
# Create object
sig = fmod.FMC(smap = smap_input, smapplt = smapplt_input, scint = scintillator)


# ----------------------------------------------------------------------------
# %% Settings
# ----------------------------------------------------------------------------
# Magnetic field at probe head, and ion species parametes (alphas here)
B, A, Z = 4, 4, 2
# pinhole and wetted area, in the same units. usually mm^2  
p_area = 3
w_area = 20000
# Fast ion distribution 
FIdist = '/shares/departments/AUG/users/alrevi/ScintSuite/MyRoutines/ITERFILD/ASCOT/ascot_dist_FILD_R8.31_FO.dat'
# Read fast ion distribution
pin_distro = fmod.read_distribution(FIdist, 
                                    pinhole_area = p_area, 
                                    wetted_area = w_area,
                                    B=B, A=A, Z=Z,
                                    version = '5.5')
# Grid definition, for weight function
PH_params = {'xmin': 20,'xmax': 90,'dx': 1,
             'ymin': 1,'ymax': 10,'dy': 0.2,} # pinhole
SC_params = {'xmin': 20,'xmax': 90,'dx': 0.5,
             'ymin': 1,'ymax': 10,'dy': 0.1,} # scintillator


# ----------------------------------------------------------------------------
# %% Generate a synthetic signal in the velocity space
# ----------------------------------------------------------------------------
plt.close('all')
# Call the synthetic signal method
sig.synthsig_pr(distro = pin_distro, mode = 'ions', 
                pin_params=PH_params, sci_params=SC_params)
# Plot the fast ion distribution
sig.plot_distribution()


# ----------------------------------------------------------------------------
# %% Create an image in the scintillator
# ----------------------------------------------------------------------------
plt.close('all')
# Establish or update image parameters.
# Any none parameters in the dictionaries means they will be removed from the 
# dictionary, others will be kept.
noise_parameters = {'neutrons': 0,}
optic_parameters = {'beta':None}
camera_parameters = {'nx':1200, 'ny':900}

# Call the synthetic signal method
sig.synthsig_xy(pin_distro, mode = 'ions', 
                pin_params = PH_params, sci_params = SC_params,
                cam_params = camera_parameters,
                opt_params = optic_parameters,
                noi_params = noise_parameters,
                smoother = 0, centering = False)
# Plot the image in the scintillator real space
sig.plot_frame_scintillator(cmap = ssplt.Gamma_I(), 
                            plot_smap = True,
                            plot_scint = True,
                            )

# ----------------------------------------------------------------------------
# %% Generate a camera frame
# ----------------------------------------------------------------------------
plt.close('all')
# Contains the data from the ITER FILD
noise_parameters = {'neutrons': 0, 'broken':0.01, 'camera_neutrons':0.001}
optic_parameters = {'T':0.39289/2, 'beta':0.2, 'FoV': [0.4398, 0.5425, 33.5]}
optic_parameters['omega']=np.pi*(0.044**2) 
camera_parameters = {'px_x_size':6.5e-6, 'px_y_size':6.5e-6,
                     'nx':2560, 'ny':2160, 'range':16, 'qe':0.6,
                     'ad_gain':0.46, 'dark_noise':1, 'exposure': 0.01,
                     'readout_noise_med':2.2, 'readout_noise_rmd':2.5,}

sig.synthsig_camera(pin_distro, mode='photons', centering=True,
                pin_params=PH_params,
                sci_params=SC_params,
                cam_params=camera_parameters,
                opt_params=optic_parameters,
                noi_params=noise_parameters,
                rm_saturation=True,
                smoother=5,
                )

# Everything can be plotted afterwards
sig.plot_distribution()
sig.plot_frame_scintillator(cmap = ssplt.Gamma_I(), 
                            plot_smap = True, plot_scint = True)
sig.plot_frame_camera(cmap=ssplt.Gamma_I(),
                      plot_smap=True, plot_scint=True, plot_FoV=True)



# ----------------------------------------------------------------------------
# %% Comparison of this method with previous
# ----------------------------------------------------------------------------
plt.close('all')
# After executing the previous block:
scint_synthetic_signal_params = {
        'rmin': 1, 'rmax': 10, 'dr': 0.1,
        'pmin': 20.0, 'pmax': 90.0, 'dp': 0.5,
        }
# --- Small check with previous forward modelling function
old_pr_space = fmod2.synthetic_signal_pr(pin_distro, WF=sig.WF, plot=True)
fig, ax = plt.subplots(1,2)
pin_diff = np.abs(sig.pr_space.ph - old_pr_space['PH'])/sig.pr_space.ph*100
pin_diff.T.plot.imshow(ax=ax[0])
sci_diff = np.abs(sig.pr_space.sc - old_pr_space['SC'])/sig.pr_space.sc*100
sci_diff.T.plot.imshow(ax=ax[1])

# The relative diference is negligible --> numeric effects
xy_frame = fmod2.original_synthsig_xy(distro = pin_distro,
                smap = smap_input, smapplt=smapplt_input, scint = scintillator,
                cam_params = camera_parameters,
                optic_params = optic_parameters,
                scint_params = scint_synthetic_signal_params,
                smoother = 5,
                centering = True)
fig, ax = fmod2.plot_the_frame(frame = xy_frame, cmap=ssplt.Gamma_I(),
                cam_params = camera_parameters,
                maxval=1, plot_smap=True, plot_scint=True,)   
fig, ax = plt.subplots()
diff = np.abs(sig.frame_scintillator.tot - xy_frame['signal_frame'])/sig.frame_scintillator.tot
diff.plot.imshow(ax=ax,vmin=0, vmax=10)
# The difference is also negligible, and not due to errors in the computation 
# of the signal

# finally the camera:
camera_noise_frame = fmod2.noise_optics_camera(frame = xy_frame,
            cam_params = camera_parameters,
            optic_params = optic_parameters,
            noise_params = noise_parameters,)
fig, ax = fmod2.plot_the_frame(frame = camera_noise_frame,
            cam_params = camera_parameters, cmap=ssplt.Gamma_I())
fig, ax = plt.subplots()
diff = np.abs(sig.frame_camera.tot - camera_noise_frame['signal_frame'])/sig.frame_camera.tot
diff.plot.imshow(ax=ax,vmin=0, vmax=10)
# Difference is also negligible
