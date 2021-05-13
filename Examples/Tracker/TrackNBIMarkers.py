"""
Launch NBI markers along a defined line and track them along the orbit

Jose Rueda

The temporal implemetnation written by jrrueda will be used here, notice that
the proper library to be used is the iHIBPSIM
"""
import Lib as ss
import os
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# AUXILIAR FUNCTIONS / variables
# -----------------------------------------------------------------------------
paths = {'bin_tracker': '/afs/ipp/home/r/ruejo/ihibpsim/bin/tracker.go',
         'results_tracker': '/afs/ipp/home/r/ruejo/iHIBPsim/results/',
         'inputs_tracker': '/afs/ipp/home/r/ruejo/iHIBPsim/inputs/'
         }

# %% Settings
# --- Basic parameters:
shot = 32312   # Shot number
t0 = 0.25
Zeff = 1.0     # Zeff
diag = 'IDE'
runID = 'E-P_diagram'   # just a name for the simulations
Nions = 1     # Number of markers to launch
NBI_number = 8
# --- Grid information
grid = {'Rmin': 1.0,
        'Rmax': 2.30,
        'nR': 1000,
        'zmin': -0.90,
        'zmax': 0.90,
        'nz': 2000}
# --- Plates (or vessel)
plate = '/afs/ipp/home/r/ruejo/iHIBPsim/bin/plate.pod'
# --- Plotting flags
p2 = True   # Plot the markers toroidal position
p3 = True   # Plot orbit evolution of some of the orbits
# -------------------------------------------------------------------------
# %% Read and write magnetic field
nameBfield = 'Bfield' + runID + '.bin'
fullnameBfield = os.path.join(paths['inputs_tracker'], nameBfield)
Bfield = ss.tracker.prepare_B_field(shot, t0, **grid, diag=diag)
ss.tracker.write_field(fullnameBfield, Bfield)

# -------------------------------------------------------------------------
# --- Markers
# %% Write markers file
namedepos = 'Deposition' + runID + '.bin'
fullnamedepos = os.path.join(paths['inputs_tracker'], namedepos)
NBI = ss.dat.NBI(NBI_number)

marker = NBI.generate_tarcker_markers(Nions)

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
            'save_orbits': '.TRUE.',
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
orbits = ss.tracker.orbit_p_phi(orbits, shot=shot, diag=diag, time=t0)
# --- Plot the temporal evolution of the orbits
if p3:
    id = np.arange(len(orbits))
    ss.tracker.plot_orbit_time_evolution(orbits, id_to_plot=id)
# --- Plot the E-p diagram
# Th input and output formats are equivalent, so we can use the same for the
# input markers:
ini_mar = ss.tracker.load_orbits(fullnamedepos, counter=Nions)
ini_mar = ss.tracker.orbit_energy(ini_mar)
ini_mar = ss.tracker.orbit_pitch(ini_mar, file=fullnameBfield)
ini_mar = ss.tracker.orbit_mu(ini_mar, file=fullnameBfield)
ini_mar = ss.tracker.orbit_p_phi(ini_mar, shot=shot, diag=diag, time=t0)
E_ini = np.zeros(Nions)
E_fin = np.zeros(Nions)
P_ini = np.zeros(Nions)
P_fin = np.zeros(Nions)
t_fin = np.zeros(Nions)
for i in range(Nions):
    if ini_mar[i]['ID'] == orbits[i]['ID']:
        E_ini[i] = ini_mar[i]['E'][0]
        E_fin[i] = orbits[i]['E'][0]

        P_ini[i] = ini_mar[i]['Pphi'][0]
        P_fin[i] = orbits[i]['Pphi'][0]

        t_fin[i] = orbits[i]['time'][0]
fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(P_ini, E_ini)
plt.show()
