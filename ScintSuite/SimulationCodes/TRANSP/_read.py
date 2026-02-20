from netCDF4 import Dataset
import xarray as xr
import numpy as np

def read_profiles(fn):
    """
    Read 1D plasma input in from a CDF file.
    
    :param fn: str, file with the TRANSP simulation.
    """
    # Read the TRANSP file
    tmp =  Dataset(fn,'r')
    a = xr.Dataset()
    a['rho'] = xr.DataArray(tmp.variables['X'][0, :], dims=['rho'],
                                coords={'rho': tmp.variables['X'][0, :]})
    # Sometimes, the TRANSP simulation fisnish before the time, so we need to cut the time array
    nt, nrho = tmp.variables['X'].shape
    t = tmp.variables['TIME'][:]
    if nt < len(t):
        t = t[:nt]
    a['t'] = xr.DataArray(t, dims=['t'])
    
    a['te'] = xr.DataArray(tmp.variables['TE'][:,:]/1e3, dims=['t','rho'])
    a['ne'] = xr.DataArray(tmp.variables['NE'][:,:]*1e6/1e19, dims=['t','rho'])
    a['ni'] = xr.DataArray(tmp.variables['NI'][:,:]*1e6/1e19, dims=['t','rho'])
    a['ti'] = xr.DataArray(tmp.variables['TI'][:,:]/1e3, dims=['t','rho'])
    a['zeff'] = xr.DataArray(tmp.variables['ZEFFP'][:,:], dims=['t','rho'])
    a['omega'] = xr.DataArray(tmp.variables['OMEGA'][:,:], dims=['t','rho'])
    # Set the metadata
    a.attrs['TRANSPfile'] = fn
    a['te'].attrs['units'] = 'keV'
    a['ne'].attrs['units'] = '1e19 m^-3'
    a['ni'].attrs['units'] = '1e19 m^-3'
    a['ti'].attrs['units'] = 'keV'
    a['zeff'].attrs['units'] = ' '
    a['t'].attrs['units'] = 's'
    a['rho'].attrs['units'] = ' '
    a['t'].attrs['long_name'] = 'Time'
    a['rho'].attrs['long_name'] = 'Normalized toroidal flux'
    a['te'].attrs['long_name'] = 'Electron temperature'
    a['ne'].attrs['long_name'] = 'Electron density'
    a['ni'].attrs['long_name'] = 'Ion density'
    a['ti'].attrs['long_name'] = 'Ion temperature'
    a['zeff'].attrs['long_name'] = 'Effective charge'
    a['omega'].attrs['long_name'] = tmp.variables['OMEGA'].long_name
    a['omega'].attrs['units'] = tmp.variables['OMEGA'].units
    return a
