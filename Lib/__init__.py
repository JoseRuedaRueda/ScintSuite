"""
Main Core of the Suite

Allows to access all the diferent libraries and capabilities

Please see https://gitlab.mpcdf.mpg.de/ruejo/scintsuite for a full
documentation


@mainpage Scintillator Suite Project
"""
import os
import f90nml
import logging
# -----------------------------------------------------------------------------
# --- Custom filter to ignore some suite message if the user wants
try:
    home = os.getenv("HOME")
    file = \
        os.path.join(home, 'ScintSuite', 'Data',
                     'MyData', 'IgnoreWarnings.txt')
    to_ignore = str(f90nml.read(file)['Warnings']['warningstoignore'])

except FileNotFoundError:
    to_ignore = 'None'


class _NoParsingFilter(logging.Filter):
    def filter(self, record, to_ignore=to_ignore):
        return not record.getMessage().startswith(to_ignore)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# --- Suite Logger. Main logging element
# Itialise the root logger
logging.basicConfig()
fmt = logging.Formatter('%(name)s | %(levelname)s | %(message)s')

# Initialise a loger of aug-sfutils. This serve to avoid duplicated entries if
# we are in AUG. If we are not in AUG, the existance of a hanging not user
# child logger will made no harm
logger = logging.getLogger('aug_sfutils')
logger.setLevel(logging.ERROR)
logger.propagate = False
# Initialise the real logger for the suite
Suite_logger = logging.getLogger('ScintSuite')

if len(Suite_logger.handlers) == 0:
    hnd = logging.StreamHandler()
    hnd.setFormatter(fmt)
    Suite_logger.addHandler(hnd)
Suite_logger.setLevel(logging.DEBUG)
Suite_logger.addFilter(_NoParsingFilter())
Suite_logger.propagate = False
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# --- Add the paths directories for python
try:
    from paths_suite import paths_of_the_suite
    paths_of_the_suite()
except (ImportError, ModuleNotFoundError):
    pass
# -----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# --- Suite modules
# Custom exceptions
import Lib.errors as err

# Load data
import Lib.LibData as dat
import Lib._ELM as ELM

# Mapping
import Lib._Mapping as mapping

# Strike Maps
import Lib._StrikeMap as smap

# Simulations codes
import Lib.SimulationCodes.FILDSIM as fildsim
import Lib.SimulationCodes.FIDASIM as fidasim
import Lib.SimulationCodes.SINPA as sinpa
import Lib.SimulationCodes.Common as simcom

# Reconstructions
import Lib._Tomography as tomo

# Handle Video files
import Lib._Video. as vid
import Lib._VRT as vrt


import Lib._Parameters as par
import Lib._Plotting as plt
import Lib._TimeTrace as tt
import Lib._Utilities as extra
import Lib._FrequencyAnalysis as ftt
import Lib._Paths as p
import Lib._Machine as m
import Lib.LibTracker as tracker
import Lib._IO as io
import Lib._FastChannel as fc
import Lib._ScintillatorCharacterization as scintcharact
import Lib._GUIs as GUI
import Lib._Optics as optics
import Lib._Noise as noise
from Lib.version_suite import version, codename
__version__ = version
import Lib._CAD as cad
import Lib._SideFunctions as side


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

# # -------------------------------------------------------------------------
# # --- Warnings handling
# # -------------------------------------------------------------------------
# # Print the ignored module warnings
# # Filter the user desired warning
# try:
#     file = \
#         os.path.join(paths.ScintSuite, 'Data',
#                      'MyData', 'IgnoreWarnings.txt')
#     to_ignore = f90nml.read(file)['Warnings']
#     print(to_ignore)
#     read_file = True
# except FileNotFoundError:
#     to_ignore = {'MissingNonEssentialModule': False}
#     read_file = False
# if not to_ignore['MissingNonEssentialModule']:
#     print(len(w))
#     for i in range(len(w)):
#         print(w[i].category)
#         if w[i].category == 'MissingNonEssentialModule':
#             print(w)
# Reset the warning filter
# # Filter the user desired warning
# if read_file:
#     if to_ignore['OverWrittingData']:
#         warnings.filterwarnings('ignore', category=war.OverWrittingData)
#     if to_ignore['MissingNonEssentialModule']:
#         warnings.filterwarnings('ignore', category=war.NotInstalledModule)
