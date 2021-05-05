"""
Launch NBI markers along a defined line and track them along the orbit

Jose Rueda

The temporal implemetnation written by jrrueda will be used here, notice that
the proper implemetnation to be used is the iHIBPSIM
"""
import Lib as ss
import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# AUXILIAR FUNCTIONS / variables
# -----------------------------------------------------------------------------
paths = {'bin_tracker': '/afs/ipp/home/r/ruejo/iHIBPsim/bin/tracker.go',
         'results_tracker': '/afs/ipp/home/r/ruejo/iHIBPsim/results/K01',
         'inputs_tracker': '/afs/ipp/home/r/ruejo/iHIBPsim/inputs/K01'
         }

# %% Settings
# --- Basic parameters:
shot = 19913   # Shot number
t0 = 4.2
Zeff = 1.0     # Zeff
runID = 'example_tracker'   # just a name for the simulations
Nions = 20     # Number of markers to launch
NBI_number = 7
# --- Grid information
grid = {'Rmin': 1.0,
        'Rmax': 2.30,
        'nR': 260,
        'zmin': -0.90,
        'zmax': 0.90,
        'nz': 360}
# --- Plates (or vessel)
plate = '/afs/ipp/home/r/ruejo/iHIBPsim/bin/plate.pod'
# --- Plotting flags
p2 = True   # Plot the markers toroidal position
p3 = True   # Plot orbit evolution
# -------------------------------------------------------------------------
# %% Read and write magnetic field
nameBfield = 'Bfield' + runID + '.bin'
fullnameBfield = os.path.join(paths['inputs_tracker'], nameBfield)
Bfield = ss.tracker.prepare_B_field(shot, t0, **grid)
ss.tracker.write_field(fullnameBfield, Bfield)

# -------------------------------------------------------------------------
# --- Markers
# %% Write markers file
namedepos = 'Deposition' + runID + '.bin'
fullnamedepos = os.path.join(paths['inputs_tracker'], namedepos)
NBI = ss.dat.NBI(NBI_number)
NBI.coords = {
    'phi0': 0.,
    'phi1': 1.,
    'x0': -1.4835,
    'y0': 0.2217,
    'z0': 1.32,
    'x1': -1.3134,
    'y1': -0.5047,
    'z1': 0.234
}
marker = ss.tracker.generate_NBI_markers(Nions, NBI)

ss.tracker.write_markers(fullnamedepos, marker)

# --- Output files
nameoutput = 'Output' + runID + '.bin'
fullnameoutput = os.path.join(paths['results_tracker'], nameoutput)

nameorbits = 'Orbits' + runID + '.bin'
fullnameorbits = os.path.join(paths['results_tracker'], nameorbits)
# -------------------------------------------------------------------------
# %% Write namelist
namelist = str(shot) + str(int(t0*100)) + runID + '.cfg'
fullnamelist = os.path.join(paths['inputs_tracker'], namelist)
opt_name = {'Bfield_name': fullnameBfield,
            'Efield_on': '.FALSE.',
            'Efield_name': '',
            'Zeff': 1.0,
            'dt': 5.0e-10,
            'max_step': 10000000,
            'Nmarkers': Nions,
            'depos_file': fullnamedepos,
            'triangle_file': plate,
            'file_out': fullnameoutput,
            'file_orbits': fullnameorbits,
            'dt_orbit': 1.0e-8}

ss.tracker.write_tracker_namelist(fullnamelist, **opt_name)
# -------------------------------------------------------------------------
# %% run the tracker
os.system(paths['bin_tracker'] + ' ' + fullnamelist)
# -------------------------------------------------------------------------
# %% Read the orbits
orbits = ss.tracker.load_orbits(fullnameorbits, counter=Nions)
# --- Calculate its energy
orbits = ss.tracker.orbit_energy(orbits)
# --- Calculate their pitch
orbits = ss.tracker.orbit_pitch(orbits, file=fullnameBfield)
# --- Calculate their magnetic moment
orbits = ss.tracker.orbit_mu(orbits, file=fullnameBfield)
# --- Calculate their Pphi
orbits = ss.tracker.orbit_p_phi(orbits, shot=32312)
# --- Plot the temporal evolution of the orbits
if p3:
    id = np.arange(len(orbits))
    ss.tracker.plot_orbit_time_evolution(orbits, id_to_plot=id)

plt.show()
