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
    """Either the frequency of adquisition function or the time trigger
    function are not found."""

    pass
