"""
Main Core of the Suite

Allows to access all the diferent libraries and capabilities

Please see https://gitlab.mpcdf.mpg.de/ruejo/scintsuite for a full
documentation


@mainpage Scintillator Suite Project
"""
# Ignore numpy warning
import warnings
import numpy as np
with warnings.catch_warnings(record=True):
    # Add the paths direactories for python
    try:
        from paths_suite import paths_of_the_suite
        paths_of_the_suite()
    except:
        pass
    # Mapping
    import Lib.LibMap as mapping

    # Simulations codes
    import Lib.SimulationCodes.FILDSIM as fildsim
    import Lib.SimulationCodes.FIDASIM as fidasim
    import Lib.SimulationCodes.SINPA as sinpa
    import Lib.SimulationCodes.Common as simcom

    # Reconstructions
    import Lib.LibTomography as tomo

    # Load data
    import Lib.LibData as dat
    import Lib.LibVideo as vid
    import Lib.LibVRT as vrt

    import Lib.LibParameters as par
    import Lib.LibPlotting as plt
    import Lib.LibTimeTraces as tt
    import Lib.LibUtilities as extra
    import Lib.LibFrequencyAnalysis as ftt
    import Lib.LibPaths as p
    import Lib.LibMachine as m
    import Lib.LibTracker as tracker
    import Lib.LibIO as io
    import Lib.LibFastChannel as fc
    import Lib.LibScintillatorCharacterization as scintcharact
    import Lib.GUIs as GUI
    import Lib.LibOptics as optics
    import Lib.LibNoise as noise
    from Lib.version_suite import version, codename
    import Lib.LibCAD as cad
    import Lib.LibSideFunctions as side

    machine = m.machine
    paths = p.Path(machine)

    # Non tokamak independent libraries
    if machine == 'AUG':
        import Lib.SimulationCodes.iHIBPsim as ihibp

    # Delte the intermedite variables to 'clean'
    del p
    del m
    # -------------------------------------------------------------------------
    # --- PRINT SUITE VERSION
    # -------------------------------------------------------------------------
    print('-... .. . -. ...- . -. .. -.. ---')
    print('VERSION: ' + version + ' ' + codename)
    print('.-- . .-.. .-.. -.-. --- -- .')

    # -------------------------------------------------------------------------
    # --- Initialise plotting options
    # -------------------------------------------------------------------------
    # It seems that with some matplotlib instalations, this could fail, so let
    # us make just a try
    try:
        plt.plotSettings()
    except:
        print('It was not possible to initialise the plotting settings')
warnings.filterwarnings('default')
