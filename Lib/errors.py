"""
Suite Errors

Contains all the Sutie custom errors which can be raised by the ScintSuite

Jose Rueda: jrrueda@us.es
"""


# --- General exceptions
class NotImplementedError(Exception):
    """Raised if the feature is still not implemented"""

    pass


class NotValidInput(Exception):
    """
    Raised if the inputs given by the user are wrong.

    Example, asking for scale 'pepe' when calling a plotting routine.

    Example 2: if one routine required the time in second or time index and you
    do not give any of them
    """

    pass


class DatabaseError(Exception):
    """Raised if there is an error accesing the tokamak/stella database."""

    pass


# --- Logbook related exceptions
class NotFoundCameraCalibration(Exception):
    """Camera calibration to align the scintillator is not found."""

    pass


class FoundSeveralCameraCalibration(Exception):
    """Several camera calibration to align the scintillator are found."""

    pass


class NotFoundGeomID(Exception):
    """GeomID used in a given shot not found."""

    pass


class FoundSeveralGeomID(Exception):
    """Several GeomID used in a given shot are found."""

    pass


class NotFoundAdqFreq_or_ttrig(Exception):
    """Either frequency of adquisition or trigger were not found."""

    pass


class NotLoadedPositionDatabase(Exception):
    """Position database was not loaded."""

    pass


# --- SINPA/FILDSIM related exception
class WrongNamelist(Exception):
    """
    Raised when some namelist parameter are not consistent

    For example when you ask for 6 gyroradius but just give 6 of them
    """

    pass


# --- Video related exception
class NoFramesLoaded(Exception):
    """Raised when you try to do something with the frames before loading   """

    pass


# --- StrikeMap
class NotFoundStrikeMap(Exception):
    """Raised if the no StrikeMap is found in the database"""

    pass


# --- StrikeMap
class NotFoundVariable(Exception):
    """Raised if the no StrikeMap is found in the database"""

    pass

# --- Data not loaded into a class
class NotDataPreloaded(Exception):
    """
    Raised, typically, within a class when the user did not load a variable
    prior to its use.

    Example : asking to plot data, but the user never gave the correspoding data.
    Example 2: asking for applying a calibration, but the user never specified one.
    """

    pass

class InconsistentData(Exception):
    """
    Raised when two sets of data with inconsistencies are used.
    """

    pass