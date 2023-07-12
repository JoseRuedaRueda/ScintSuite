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
import shutil
## ----------------------------------------------------------------------------
# --- Filters and color handler, logging

# home = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
home = os.path.join(os.getenv("HOME"), 'ScintSuite')
print(home)
try:
    file = \
        os.path.join(home, 'Data',
                     'MyData', 'IgnoreWarnings.txt')
    to_ignore = str(f90nml.read(file)['Warnings']['warningstoignore'])
except FileNotFoundError:
    file_template = \
        os.path.join(home, 'Data', 'MyDataTemplates', 'IgnoreWarnings.txt')
    shutil.copyfile(file_template, file)
    to_ignore = 'None'

# Checking if paths and plot files are also in the MyData.
file = os.path.join(home, 'Data',
                    'MyData', 'Paths.txt')
if not os.path.isfile(file):
    file_template = \
        os.path.join(home, 'Data', 'MyDataTemplates', 'Paths.txt')
    shutil.copyfile(file_template, file)

file = os.path.join(home, 'Data',
                    'MyData', 'plotting_default_param.cfg')
if not os.path.isfile(file):
    file_template = \
        os.path.join(home, 'Data', 'MyDataTemplates', 'plotting_default_param.cfg')
    shutil.copyfile(file_template, file)

class _NoParsingFilter(logging.Filter):
    def filter(self, record, to_ignore=to_ignore):
        return not record.getMessage().startswith(to_ignore)


class _CustomFormatter(logging.Formatter):

    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(name)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


## ----------------------------------------------------------------------------
# --- Suite Logger. Main logging element
# Itialise the root logger
logging.basicConfig()

# Initialise a loger of aug-sfutils. This serve to avoid duplicated entries if
# we are in AUG. If we are not in AUG, the existence of a hanging not user
# child logger will make no harm
logger = logging.getLogger('aug_sfutils')
if len(logger.handlers) == 0:
    hnd = logging.StreamHandler()
    hnd.setFormatter(_CustomFormatter())
    logger.addHandler(hnd)
logger.setLevel(logging.ERROR)
logger.propagate = False
# Initialise the real logger for the suite
Suite_logger = logging.getLogger('ScintSuite')

if len(Suite_logger.handlers) == 0:
    hnd = logging.StreamHandler()
    hnd.setFormatter(_CustomFormatter())
    Suite_logger.addHandler(hnd)
Suite_logger.setLevel(logging.INFO)
Suite_logger.addFilter(_NoParsingFilter())
Suite_logger.propagate = False

## ----------------------------------------------------------------------------
# --- Add the paths directories for python
try:
    from paths_suite import paths_of_the_suite
    paths_of_the_suite()
except (ImportError, ModuleNotFoundError):
    pass


## ----------------------------------------------------------------------------
# --- Suite modules
# Custom exceptions
import Lib.errors as err

# Load data
import Lib.LibData as dat
import Lib._ELM as ELM

# custom fits models
import Lib._CustomFitModels as cfm

# Mapping
import Lib._Mapping as mapping

# Strike Maps
import Lib._StrikeMap as smap

# Simulations codes
import Lib.SimulationCodes.FILDSIM as fildsim
import Lib.SimulationCodes.SINPA as sinpa
import Lib.SimulationCodes.Common as simcom
try:
    import pyLib as fidasim
except ModuleNotFoundError:
    pass

# Reconstructions
from Lib._Tomography._main_class import Tomography as tomography

# Handle Video files
import Lib._Video as vid
import Lib._VRT as vrt

# MHD activity
import Lib._MHD as mhd

# Handle ufiles
from Lib.ufiles import ufile as Ufile

# Handle Scintillators
import Lib._Scintillator as scint

import Lib._Parameters as par
import Lib._Plotting as plt
import Lib._TimeTrace as tt
import Lib._Utilities as extra
import Lib._FrequencyAnalysis as ftt
import Lib._Paths as p
import Lib._Machine as m
import Lib._IO as io
import Lib._FastChannel as fc
import Lib._GUIs as GUI
import Lib._Optics as optics
import Lib._Noise as noise
import Lib.version_suite as ver
from Lib.version_suite import version, codename
__version__ = version
__codename__ = codename
import Lib._CAD as cad
import Lib._SideFunctions as side


machine = m.machine
paths = p.Path(machine)

# Non tokamak independent libraries
if machine == 'AUG':
    import Lib.SimulationCodes.torbeam as torbeam
    import Lib.SimulationCodes.iHIBPsim as ihibp

# Delete the intermediate variables to 'clean'
del p
del m
## ------------------------------------------------------------------------
# --- PRINT SUITE VERSION
# -------------------------------------------------------------------------
logger.info('-... .. . -. ...- . -. .. -.. ---')
logger.info('VERSION: ' + version + ' ' + codename)
logger.info('.-- . .-.. .-.. -.-. --- -- .')
ver.printGITcommit()
## ------------------------------------------------------------------------
# --- Initialise plotting options
# -------------------------------------------------------------------------
# It seems that with some matplotlib installations, this could fail, so let
# us make just a try
try:
    plt.plotSettings()
except:
    logger.warning('28: It was not possible to initialise the plotting ' +
                   'settings')
