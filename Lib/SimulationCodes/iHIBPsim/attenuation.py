"""Contains the methods and classes to interact with iHIBPsim
 - Beam attenuation"""

import os
import numpy as np
import matplotlib.pyplot as plt
import Lib._Plotting as ssplt
from scipy.interpolate import interp1d, interp2d
try:
    import netCDF4 as nc4
except ModuleNotFoundError:
    print('netCDF4 library not found. Install it to use iHIBPsim.')


# --------------------------------------------------------
# --- Functions to read the tables.
# --------------------------------------------------------
def loadTable1D(filename: str):
    """
    Load a reaction-rate table that is based in (T, sigma(T)).

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param filename: Filename containing the table.

    """
    output = {}  # Empty dictionary.
    with open(filename, 'rb') as fid:
        nT = np.fromfile(fid, 'int32', 1)[0]
        T = np.fromfile(fid, 'float64', nT)
        sigma = np.fromfile(fid, 'float64', nT)

        output['table_type'] = 1
        output['nBase'] = nT
        output['base'] = T
        output['sigma'] = sigma
        output['base_name'] = 'Temperature'
        output['base_units'] = 'eV'
        output['units'] = '$m^3/s$'

        # --- Creating the interpolating object.
        output['interp'] = interp1d(np.log(output['base']),
                                    np.log(output['sigma']),
                                    kind='linear',
                                    fill_value='extrapolate',
                                    assume_sorted=True,
                                    bounds_error=False)

    return output


def loadTable2D(filename: str):
    """
    Load a reaction-rate table that is based in (T,v, sigma(v,T)).

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param filename: Filename containing the table.
    """
    output = {}  # Empty dictionary.
    with open(filename, 'rb') as fid:
        nT = np.fromfile(fid, 'int32', 1)[0]
        nv = np.fromfile(fid, 'int32', 1)[0]
        T = np.fromfile(fid, 'float64', nT)
        v = np.fromfile(fid, 'float64', nv)
        sigma = np.fromfile(fid, 'float64', nT*nv).reshape((nv, nT), order='F')
        output['table_type'] = 2
        output['nBase'] = []
        output['nBase'].append(nT)
        output['nBase'].append(nv)
        output['base'] = []
        output['base'].append(T)
        output['base'].append(v)
        output['sigma'] = sigma
        output['base_units'] = []
        output['base_units'].append('eV')
        output['base_units'].append('m/s')
        output['base_name'] = []
        output['base_name'].append('Temperature')
        output['base_name'].append('Velocity')
        output['base_shortname'] = []
        output['base_shortname'].append('T')
        output['base_shortname'].append('v')
        output['units'] = '$m^3/s$'

        # --- Creating the interpolating object.
        output['interp'] = interp2d(np.log(output['base'][0]),
                                    output['base'][1],
                                    np.log(output['sigma']),
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value='extrapolate')

    return output


# --------------------------------------------------------
# --- Functions to plot the tables.
# --------------------------------------------------------
def plotTable1D(table: dict, ax=None, ax_options: dict = {},
                line_options: dict = {}, grid: bool = True,
                label: str = None, loglog: bool = True):
    """
    Plotting the ionization table. The input is provides as the output.
    of 'loadTable1D'.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param table: dictionary with the data to be plotted.
    """
    # --- Creating axis and checking the plotting inputs.
    axis_was_none = False
    if ax is None:
        axis_was_none = True
        fig, ax = plt.subplots(1, 1)

    # --- Initialise the plotting parameters
    # if 'fontsize' not in ax_options:  # Deprecated, is not done via the
    #     ax_options['fontsize'] = 16   # default plotting settings
    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'
    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2

    # --- Checking input table.
    ax_options['xlabel'] = ''
    if 'name' in table:
        ax_options['title'] = table['name']
    if 'base_name' in table:
        ax_options['xlabel'] += table['base_name']
    if 'base_units' in table:
        ax_options['xlabel'] += ' ['+table['base_units']+']'

    ax_options['ylabel'] = ''
    if 'short_name' in table:
        ax_options['ylabel'] += table['short_name']
    if 'units' in table:
        ax_options['ylabel'] += ' ['+table['units']+']'

    if loglog and axis_was_none:
        ax_options['xscale'] = 'log'
        ax_options['yscale'] = 'log'

    ax.plot(table['base'], table['sigma'], **line_options,
            label=label)

    if grid and axis_was_none:
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')

    ax = ssplt.axis_beauty(ax, ax_options)
    plt.tight_layout()

    return ax


def plotTable2D(table: dict, v: float = None, ax=None, ax_options: dict = {},
                line_options: dict = {}, grid: bool = True,
                loglog: bool = True):
    """
    Plot the ionization table.

    The input is provides as the output of 'loadTable1D'.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param table: dictionary with the data to be plotted.
    """
    # --- Creating axis and checking the plotting inputs.
    axis_was_none = False
    if ax is None:
        axis_was_none = True
        fig, ax = plt.subplots(1, 1)

    # --- Initialise the plotting parameters
    # if 'fontsize' not in ax_options:  # Deprecated, is not done via the
    #     ax_options['fontsize'] = 16   # default plotting settings
    if 'grid' not in ax_options:
        ax_options['grid'] = 'both'
    if 'linewidth' not in line_options:
        line_options['linewidth'] = 2

    # --- Checking input table.
    ax_options['xlabel'] = ''
    if 'name' in table:
        ax_options['title'] = table['name']
    if 'base_name' in table:
        ax_options['xlabel'] += table['base_name'][0]
    if 'base_units' in table:
        ax_options['xlabel'] += ' ['+table['base_units'][0]+']'

    ax_options['ylabel'] = ''
    if 'short_name' in table:
        ax_options['ylabel'] += table['short_name']
    if 'units' in table:
        ax_options['ylabel'] += ' ['+table['units']+']'

    if loglog:
        ax_options['xscale'] = 'log'
        ax_options['yscale'] = 'log'
    else:
        ax_options['xscale'] = 'linear'
        ax_options['yscale'] = 'linear'

    if v is None:
        v = table['base'][1]

    # --- Interpolating the tables to the input velocities.
    TT, VV = np.meshgrid(table['base'][0], v)
    sigma = table['interp'](TT, VV).reshape(VV.shape)
    print(str(sigma.shape))

    # --- Plotting the reaction-rates.
    for ii in range(len(v)):
        ax.plot(table['base'][0], sigma[ii, :], **line_options,
                label=table['base_shortname'][1]+'='+str(v[ii]))

    if grid and axis_was_none:
        ax.grid(True, which='minor', linestyle=':')
        ax.minorticks_on()
        ax.grid(True, which='major')

    ax = ssplt.axis_beauty(ax, ax_options)
    plt.tight_layout()

    return ax


# --------------------------------------------------------
# --- Functions to read from the database in Python.
# --------------------------------------------------------
def read_rates_Database(name: str = 'Cs',
                        filename: str = 'Data/Tables/iHIBP/rates.cdf'):
    """
    Read the reaction rates of a given species from the database.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param name: name of the species to be read from the database.
    @param filename: name of the netCDF4 file containing the data.
    """
    try:
        fid = nc4.Dataset(filename, mode='r')
    except:
        raise Exception('No file was found!')

    try:
        # For each species, we have a different group.
        rate_group_name = '/'+name.lower()+'/'
        rate_group = fid.groups[rate_group_name]
    except:
        raise Exception('The species has not the rate stored in the database')
        fid.close()

    try:
        # We get the number of rates stored into the database.
        num_reactions = rate_group['num_reactions']

        # --- Opening the dictionary:
        output = dict()

        output['species'] = rate_group.species
        output['date'] = rate_group.date
        output['species_symbol'] = rate_group.symbol
        output['rates'] = []

        # --- Getting the reactions:
        for ii in range(num_reactions):
            reaction_g = rate_group.groups['reaction_'+str(ii)]
            rate = dict()
            rate['atom'] = dict()
            rate['base'] = []

            # Reading properties.
            rate['atom']['name'] = reaction_g.name
            rate['atom']['symbol'] = reaction_g.symbol
            rate['atom']['binding_energy'] = reaction_g['binding_energy']
            rate['atom']['binding_units'] = reaction_g.binding_units
            rate['reaction'] = reaction_g.reaction
            rate['reaction_name'] = reaction_g.reaction_name

            # Reading the basis data for the rates.
            numBasis = reaction_g.num_basis
            for jj in range(numBasis):
                base_name = 'base' + str(jj)
                base = reaction_g[base_name]

                rate['base'].append({'data': base[:],
                                     'units': base.units,
                                     'short': base.short,
                                     'name': base.long
                                     })

            # Reading the reaction rate.
            rate['rate'] = reaction_g['data'][:]
            rate['units'] = reaction_g['data'].units

            output.append(rate)
            del rate
    except:
        raise Exception('Error while reading the data')

    fid.close()
    return output


def write_rates_Database(rates: dict, name: str = 'Cs',
                         filename: str = 'Data/Tables/iHIBP/rates.cdf'):
    """
    Write to a netCDF4 file the DB of reaction rates.

    Pablo Oyola - pablo.oyola@ipp.mpg.de

    @param rates: dictionary with all the information to write
    to the file.
    @param name: name of the species to save in the datafile.
    @param filename: location of the datafile to write the file. If the file
    does not exist, this routine will try to create one.
    """
    # --- Checking if the file exists:
    if not os.path.isfile(filename):
        fid = nc4.Dataset(filename, 'w')  # Creating a new file
    else:
        fid = nc4.Dataset(filename, 'a')  # Appending to the file

    rate_g = fid.createGroup('/'+name.lower()+'/')

    rate_g.createDimension('num_reactions', size=rate_g)
    rate_g.date = rates['date']
    rate_g.species = rates['species']
    rate_g.symbol = rates['species_symbol']

    for ii in range(len(rates['rates'])):
        reaction_g = rate_g.createGroup('reaction_'+str(ii))

        # Writing the properties.
        reaction_g.name = rates['atom']['name']
        reaction_g.symbol = rates['atom']['symbol']
        reaction_g.binding_units = rates['atom']['binding_units']
        reaction_g.reaction = rates['reaction']
        reaction_g.reaction_name = rates['reaction_name']

        # Writing the basis data.
        reaction_g.createDimension('numBasis',
                                   size=len(rates['rate'][ii]['base']))

        basis_list = []
        for jj in len(rates['rate'][ii]['base']):
            base_name = 'base' + str(jj)
            base_size = len(rates['rate'][ii]['base'][jj]['data'])

            # Creating the dimension variable
            reaction_g.createDimension(base_name+'_size',
                                       size=base_size)

            base = reaction_g.createVariable(base_name, np.float64,
                                             dimensions=(base_name+'_size',))

            base[:] = rates['rate'][ii]['base'][jj]['data']
            base.units = rates['rate'][ii]['base'][jj]['units']
            base.long = rates['rate'][ii]['base'][jj]['name']
            base.short = rates['rate'][ii]['base'][jj]['short']
            base.useful = rates['rate'][ii]['base'][jj]['useful']

            if rates['rate'][ii]['base'][jj]['useful']:
                basis_list.append(base_name+'_size')

        reaction_data = reaction_g.createVariable('data', np.float64,
                                                  dimensions=basis_list)
        reaction_data[:] = rates['rate'][ii]['rate']
        reaction_data.units = rates['rate'][ii]['units']
