"""
Main Core of the Suite

Allows to access all the diferent libraries and capabilities

Please see https://gitlab.mpcdf.mpg.de/ruejo/scintsuite for a full
documentation


@mainpage Scintillator Suite Project
"""
import os
import yaml
import logging
import shutil

# -----------------------------------------------------------------------------
# %% Read the settings file
# -----------------------------------------------------------------------------
home = os.getenv("ScintSuitePath")
if home is None:
    home = os.path.join(os.getenv("HOME"), 'ScintSuite')
UserSettings = os.path.join(home, 'Settings.yml')
with open(UserSettings, 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise Exception('Error reading the settings file')
# Get the warnings to ignore:
to_ignore = settings['warningsToIgnore']

# ----------------------------------------------------------------------------
# %% Set the logger
# ----------------------------------------------------------------------------
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
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Initialise the real logger for the suite
Suite_logger = logging.getLogger('ScintSuite')

if len(Suite_logger.handlers) == 0:
    hnd = logging.StreamHandler()
    hnd.setFormatter(_CustomFormatter())
    Suite_logger.addHandler(hnd)
Suite_logger.setLevel(logging.DEBUG)
Suite_logger.addFilter(_NoParsingFilter())
Suite_logger.propagate = False

# ----------------------------------------------------------------------------
# %% Extend the path to find machine dependent libraries
# -----------------------------------------------------------------------------
try:
    from envPathExtend import envPathExtend
    envPathExtend()
except (ImportError, ModuleNotFoundError):
    pass


# ----------------------------------------------------------------------------
# %% Suite modules
# Custom exceptions
import ScintSuite.errors as err

# Load data
import ScintSuite.LibData as dat
import ScintSuite._ELM as ELM

# custom fits models
import ScintSuite._CustomFitModels as cfm

# Mapping
import ScintSuite._Mapping as mapping

# Strike Maps
import ScintSuite._StrikeMap as smap

# Simulations codes
import ScintSuite.SimulationCodes.FILDSIM as fildsim
import ScintSuite.SimulationCodes.SINPA as sinpa
import ScintSuite.SimulationCodes.Common as simcom
import ScintSuite.SimulationCodes.TRANSP as transp
import ScintSuite.SimulationCodes.OWCF as OWCF
from ScintSuite._MULTIPOW import MULTIPOW

try:
    import FIDASIM4py as fidasim
except ModuleNotFoundError:
    pass

# Reconstructions
try:
    from ScintSuite._Tomography._main_class import Tomography as tomography
except:
    pass

# Handle Video files
import ScintSuite._Video as vid
import ScintSuite._VRT as vrt

# MHD activity
import ScintSuite._MHD as mhd

# Handle ufiles
from ScintSuite.ufiles import ufile as Ufile

# Handle Scintillators
import ScintSuite._Scintillator as scint

import ScintSuite._Parameters as par
import ScintSuite._Plotting as plt
import ScintSuite._TimeTrace as tt
import ScintSuite._Utilities as extra
import ScintSuite._FrequencyAnalysis as ftt
import ScintSuite._Paths as p
import ScintSuite._Machine as m
import ScintSuite._IO as io
import ScintSuite._FastChannel as fc
import ScintSuite._GUIs as GUI
import ScintSuite._Optics as optics
import ScintSuite._Noise as noise
import ScintSuite.version_suite as ver
from ScintSuite.version_suite import version, codename
from ScintSuite._pySpecView import pySpecView
__version__ = version
__codename__ = codename
import ScintSuite._CAD as cad
import ScintSuite._SideFunctions as side


machine = m.machine
paths = p.Path(machine)

# Non tokamak independent libraries
if machine == 'AUG':
    import ScintSuite.SimulationCodes.torbeam as torbeam

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
