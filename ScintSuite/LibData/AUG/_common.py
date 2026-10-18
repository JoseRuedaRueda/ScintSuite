"""
Common utilities, decorators, and safe shotfile loading helpers for AUG misc.
"""

import os
import numpy as np
import xarray as xr
xr.set_options(keep_attrs=True)

import aug_sfutils as sf
import logging
logger = logging.getLogger("AUG_data")


def safe_sfread(diag: str, shot: int, experiment: str = 'AUGD', 
                edition: int = 0, **kwargs):
    """
    Safely attempts to open an ASDEX Upgrade shotfile using aug_sfutils
    If the shotfile does not exist, fails to load, or sf is not installed,
    it catches the exception, logs at debug level, and returns None

    :param diag: Diagnostic name
    :param shot: Shot number
    :param experiment: Experiment name
    :param edition: Edition of the shotfile
    :return: Shotfile reader object, or None on failure
    """
    if sf is None or shot is None or diag is None:
        return None
    try:
        sfo = sf.SFREAD(diag, int(shot), experiment=experiment, edition=edition, **kwargs)
        if sfo is not None and getattr(sfo, 'status', True):
            return sfo
    except Exception:
        pass
    try:
        sfo = sf.SFREAD(int(shot), diag, experiment=experiment, edition=edition, **kwargs)
        if sfo is not None and getattr(sfo, 'status', True):
            return sfo
    except Exception:
        pass

    return None

def safe_equ(shot: int, diag: str = 'EQH', exp: str = 'AUGD', ed: int = 0, **kwargs):
    """
    Safely creates an Equilibrium (EQU) object from aug_sfutils.
    Returns the EQU object if successful, or None on failure.

    :param shot: Shot number
    :param diag: Equilibrium diagnostic (default 'EQH')
    :param exp: Experiment (default 'AUGD')
    :param ed: Edition (default 0)
    :return: EQU object or None
    """
    if sf is None or shot is None:
        return None

    try:
        equ = sf.EQU(int(shot), diag=diag, exp=exp, ed=ed, **kwargs)
        if equ is not None and getattr(equ, 'status', True):
            return equ
    except Exception:
        pass

    return None

def _clean_val(v):
    """
    Converts numpy scalar objects to native Python values
    """
    if hasattr(v, "item") and np.ndim(v) == 0:
        return v.item()
    return v


def to_dict_with_metadata(obj):
    """
    Converts xarray Dataset or DataArray into a dictionary preserving metadata
    
    Returns None if obj is None
    """
    if obj is None:
        return None
    if not isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj

    output = {}
    for k, v in obj.attrs.items():
        output[k] = _clean_val(v)

    if isinstance(obj, xr.Dataset):
        for var_name, var_da in obj.data_vars.items():
            output[var_name] = var_da.values
            for attr_k, attr_v in var_da.attrs.items():
                output[f"{var_name}_{attr_k}"] = _clean_val(attr_v)
    else:  # DataArray
        output["data"] = obj.values

    for coord_name, coord_da in obj.coords.items():
        output[coord_name] = coord_da.values
        for attr_k, attr_v in coord_da.attrs.items():
            output[f"{coord_name}_{attr_k}"] = _clean_val(attr_v)

    return output
