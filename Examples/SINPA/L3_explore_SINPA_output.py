"""
Lecture 3 of the examples to run the SINPA code: Analysis

Jose Rueda Rueda: jrrueda@us.es

Done to explain how to explore and analysise SINPA runs

It is supposed that you have run your SINPA simulation before

Created for version 0.7.3 of the Suite and version 0.1 of SINPA
"""
import os
import Lib as ss
import matplotlib.pyplot as plt

# --- Settings
runID = 'Example'  # runID of the performed SINPA simulation
smap_flag = True  # execute the strike_map demostration block
orbits_flag = True  # execute the orbit demostration block
collimator_flag = True  # execute the collimator demostration block


# --- StrikeMap demostration block
if smap_flag:
    smap_name = \
        os.path.join(ss.paths.SINPA, 'runs', runID, 'results', runID + '.map')
    smap = ss.mapping.StrikeMap('FILD', file=smap_name)
    smap.load_strike_points()
    # Notice that by default, the method which loads the strike points
    # calculate and apply the interpolators, as this is no longer done inside
    # fortran for the SINPA code. If you need to change the used interpolators,
    # you can do it with the followed functions:ç
    # smap.calculate_mapping_interpolators(): it calculate the interpolators,
    # accept several options to change the kernel and degree of the function
    # smap.remap_strike_points(): just apply the interpolator to the strike
    # points and store the information inside the matrix
    # smap.strike_points.data, it also update the information of the header of
    # the strike points object

    # plot the strike map
    smap.plot_real(labels=True)
    # plot it but with a different scale (notice that the smap is now in m! if
    # the user want to plotted in cm like in the old FILDSIM, the next line
    # will do the job):
    smap.plot_real(factor=100.0)

    # now for example let's plot the strike points position from the 2nd and
    # 4th pitch angle:
    fig, ax = plt.subplots()
    smap.strike_points.scatter(XI_index=[1, 3], ax=ax)
    # overplot the strike map
    smap.plot_real(ax=ax)

    # we can use the same method to plot any 2 variables of the strike points
    # for example, we can explore the correlations between the initial
    # gyrophase (called beta in the suite) and the incident angle (called theta
    # in the suite)
    smap.strike_points.scatter(varx='beta', vary='theta')

    # Now for example let's calculate the FILD resolutions
    smap.calculate_resolutions()
    # And let's plot them
    smap.plot_resolutions()

    # Plot also the fits done for the calculation (notice, there are many plot
    # style, explore your beloved one)
    smap.plot_resolution_fits(var='Gyroradius',
                              XI_index=3,
                              gyroradius=None, pitch=None,
                              kind_of_plot='normal',
                              include_legend=True)
    plt.show()

# --- Orbits demostration block
if orbits_flag:
    # Load the orbits
    orb = ss.sinpa.orbits(runID='Example')
    # return just one particular orbit
    particular_orbit = orb[1]  # Just an example, I'll do nothing with it
    # Plot them in 3D:
    # plot in red the ones colliding with the scintillator
    orb.plot3D(line_params={'color': 'r'}, kind=(2,))
    # plot in blue the ones colliding with the collimator
    ax = plt.gca()
    orb.plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax)
    # Plot the geometry on top
    Geom = ss.simcom.Geometry(GeomID='FILD1')
    Geom.plot3Dfilled(ax=ax, units='m')
    # plot the 'wrong orbits', to be sure that the are just orbits exceeding
    # the scintillator and nothing is wrong with the code
    orb.plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax)
