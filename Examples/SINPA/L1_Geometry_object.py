"""
Lecture 1 of the General introduction to the SINPA code.

Import, use and plot of the geometry object

Jose Rueda Rueda - jrrueda@us.es
Alex Reyner Viñolas - areyner@us.es

Note: written for the version 0.7.3 of the Scintillator Suite and the version
0.1 of the SINPA code

Note 2: the geometry object can also load and handle FILDSIM geometries, so all
the plotting and handling done after loading the stuff is similar. To see how
to load FILDSIM geometry, see the documentation of the object

last revision:
    ScintSuite: version 2.0.1
    SINPA (uFILDSIM): version 2.3
"""
import ScintSuite as ss
import ScintSuite.SimulationCodes.Common.geometry as ssgeo


## --- Reduce
# This functions allows you to reduce an .stl file to a certain number of triangles
# If triangles are to low, the function starts to interpolate the surface
# Useful for over resolved geometries that do not need as much precision, and
# sometimes easier than doing it by hand in CAD or other softwares 
ssgeo.decimate_stl(path = 'path/to/stl/file', triangles = 20)
# Output has the same name with _dec.stl at the end


## --- Import geometry from .stl
# This functions get your .stl files and transform them into the triangle files
# stored in /SINPA/Geometry. You can add as many files as you want.
# The pinhole will be placed, as per default, in (0.04, 0, 0) m. It will have a 
# size of (1, 2, 0) mm.
# A good recomendation is editing your files before, and align them to this.
ssgeo.stl2geometry(
    geomID = 'Test0',
    scintillator_stl_files = {
        'scintillator': '/path/to/stl/scintillator.stl'
    },
    collimator_stl_files = {
        'collimator': '/path/to/stl/collimator.stl',
        'heatshield': '/path/to/stl/heatshield.stl',
    },
)

## --- Edit geometry
# With this function you can move the full geometry around.
# Operates over the triangle files, and edit the ExtraGeometryParams accordingly
geomID = 'Test0'
new_geomID = 'Test1'
ssgeo.edit_geometry(
    geomID, new_geomID, 
    shift = [0.1, 0.2, 0.3], # vector to shift the geometry. In m
    mult = 1, # scaling factor
    ax = 'x', # in which axis the geometry needs to be mirrored? x, y, z, xy...
    relocate = True, # keep the pinhole en the same position (after shifting)
    inversion = False, # sometime the normal of the triangles can be wrong. reverses that
    ignore = None # if you want any element to be ignored, write it here
)


## --- Plot the geometry
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
