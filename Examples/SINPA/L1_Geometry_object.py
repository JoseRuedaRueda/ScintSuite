"""
Lecture 1 of the General introduction to the SINPA code.

Use and plot of the geometry object

Jose Rueda Rueda - jrrueda@us.es

Note: written for the version 0.7.3 of the Scintillator Suite and the version
0.1 of the SINPA code

Note 2: the geometry object can also load and handle FILDSIM geometries, so all
the plotting and handling done after loading the stuff is similar. To see how
to load FILDSIM geometry, see the documentation of the object

last revision:
    ScintSuite: version 1.0.4
    SINPA (uFILDSIM): version 2.3
"""
import ScintSuite.as ss

# --- Settings
geom_ID = 'FILD1'

# --- Load the geometry
# To use this import method, the folder containing the elements should be
# inside the SINPA folder. If you have a custom path, please explore the optinal
# argument of the ss.simcom.Geometr() object
Geometry = ss.simcom.Geometry(GeomID=geom_ID)
# Trivia note, simcom comes from SimulationCommon. In that submodule you can
# find the object which are common from FILDSIM/SINPA, and even iHIBPSIM,
# the fields

# --- Examples of things which can be done
# Print the number of elements in the geometry:
print('Your genemetry has ', Geometry.size, 'elements')

# Plot the geometry as a series of lines, in 3D. But just the scintillator
Geometry.plot3Dlines(element_to_plot=[2])
# now collimator + scintillator
Geometry.plot3Dlines(element_to_plot=[0, 2])
# now in meters instead of cm
Geometry.plot3Dlines(element_to_plot=[0, 2], units='m')

# Now in 3D with shaded object (as before you can select the plate you want to
# to plot)

Geometry.plot3Dfilled(element_to_plot=[0, 2])
