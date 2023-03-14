"""
Lecture 3 of the examples to run the SINPA code: Analysis

Jose Rueda Rueda: jrrueda@us.es

Done to explain how to explore and analysise SINPA runs

It is supposed that you have run your SINPA simulation before

Created for version 0.7.3 of the Suite and version 0.1 of SINPA

last revision:
    ScintSuite: version 1.0.4
    SINPA (uFILDSIM): version 2.3
"""
import os
import Lib as ss
import matplotlib.pyplot as plt

# --- Settings
runID = 'MU01_map_-016.00000_010.00000'  # runID of the performed SINPA simulation
smap_flag = False  # execute the strike_map demonstration block
orbits_flag = True  # execute the orbit demonstration block
collimator_flag = True  # execute the collimator demonstration block


# --- StrikeMap demostration block
if smap_flag:
    # Guess the name of the map based on the runID
    smap_name = \
        os.path.join(ss.paths.SINPA, 'runs', runID, 'results', runID + '.map')
    # Load the map
    smap = ss.smap.Fsmap(file=smap_name)
    # Load the strike points
    smap.load_strike_points()
    # Notice that by default, the method which loads the strike points
    # calculate and apply the interpolators, as this is no longer done inside
    # fortran for the SINPA code. If you need to change the used interpolators,
    # you can do it with the followed functions:ç
    # smap._calculate_mapping_interpolators(): it calculate the interpolators,
    # accept several options to change the kernel and degree of the function
    # smap.remap_strike_points(): just apply the interpolator to the strike
    # points and store the information inside the matrix smap.strike_points.data
    # you can remap in the pixel or in the real space it also update the
    # information of the header of he strike points object

    # plot the strike map
    smap.plot_real(labels=False)
    # Labels are a bit of a mesh as we have to conciliate several FILDS with
    # several options, updates are more than wellcome

    # plot it but with a different scale (notice that the smap is now in m! if
    # the user want to plotted in cm like in the old FILDSIM, the next line
    # will do the job):
    smap.plot_real(factor=100.0, labels=False)

    # now for example let's plot the strike points position from the 2nd and
    # 4th pitch angle:
    fig, ax = plt.subplots()
    smap.strike_points.scatter(varx='x1', vary='x2', XI_index=[1, 3], ax=ax)
    # Notice that XI_index select the pitch 2 and 4 of the launched, in
    # ascendent order, 'x1' and 'x2' are the 2 coorditanes of the strike points
    # on the scintillator
    # overplot the strike map
    smap.plot_real(ax=ax, labels=False)
    plt.draw()

    # we can use the same method to plot any 2 variables of the strike points
    # for example, we can explore the correlations between the initial
    # gyrophase (called beta in the suite) and the incident angle (called theta
    # in the suite)
    smap.strike_points.scatter(varx='beta', vary='theta')

    # Now for example let's calculate the FILD resolutions (with default params)
    smap.calculate_phase_space_resolution()
    # This is kinda deprecates, still use the old sintax, will be changed in the
    # next version
    smap.plot_phase_space_resolution()

    # Plot also the fits done for the calculation (notice, there are many plot
    # style, explore your beloved one)
    # And let's plot the fit on giroradius of the 3 pitch angle
    # This is kinda deprecates, still use the old sintax, will be changed in the
    # next version
    smap.plot_phase_space_resolution_fits(var='Gyroradius',
                              XI_index=3,
                              gyroradius=None, pitch=None,
                              kind_of_plot='normal',
                              include_legend=True)
    plt.show()

# --- Orbits demostration block
if orbits_flag:
    # Load the orbits
    orb = ss.sinpa.orbits(runID=runID)
    # return just one particular orbit
    particular_orbit = orb[1]  # Just an example, I'll do nothing with it
    # Plot them in 3D:
    # plot in red the ones colliding with the scintillator
    ax = orb.plot3D(line_params={'color': 'r'}, kind=(2,))
    # plot in blue the ones colliding with the collimator
    plt.gca()
    orb.plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax)
    # Plot the geometry on top
    # Note. HARDCORED GEOMETRY!!!
    # One could just read the geom ID from the namelist, but I was lazy.
    Geom = ss.simcom.Geometry(GeomID='MU01')
    Geom.plot3Dfilled(ax=ax, units='m')
    # plot the 'wrong orbits', to be sure that the are just orbits exceeding
    # the scintillator and nothing is wrong with the code
    orb.plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax)
