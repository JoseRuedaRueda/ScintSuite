"""
Read the OWCF topological map and implement basic resonance calculations.

Jose Rueda Rueda: jruedaru@uci.edu

Introduced in version 1.4.0

Adapted from the ihibsim code: https://github.com/pablo-oyola/ihibpsim/
"""
import logging
import numpy as np
import os
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.constants import elementary_charge as ec
from scipy.interpolate import RegularGridInterpolator
from typing import Union, Iterable
from copy import deepcopy
from tqdm import tqdm

logger = logging.getLogger('ScintSuite.OWCF')

from scipy.interpolate import splrep, BSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider

# ---- Allocate in the data set the variables to store the scan
RESONANCES_VARIABLES = {
            'Wpol': {
                'units': 'rad/s',
                'long_name': '$\\omega_{pol}$',
                'name': 'w_pol'
            },
            'Wtor': {
                'units': 'rad/s',
                'long_name': '$\\omega_{tor}$',
                'name': 'w_tor'
            },
            'Wdrift': {
                'units': 'rad/s',
                'long_name': '$\\omega_{b}$',
                'name': 'w_drift'
            },
            'Kind': {
                'units': '',
                'long_name': 'Kind',
                'name': 'kind'
            },
            'T': {
                'units': 's',
                'long_name': 'Integration Time',
            },
            'IDs': {
                'units': '',
                'long_name': 'Particle ID',
            },
            'MaskWtor': {
                'units': '',
                'long_name': 'Mask with physics Wtor',
                'type': bool
            },
            'MaskWpol': {
                'units': '',
                'long_name': 'Mask with physics Wpol',
                'type': bool
            },
            'MaskWdrift': {
                'units': '',
                'long_name': 'Mask with physics Wdrift',
                'type': bool
            },
            'Counter': {
                'units': '',
                'long_name': 'Sanityariable',
                'type': int
            },
        }

class TopoMap:
    """
    Class to read the topological map from the OWCF code.
    """
    def __init__(self, filename: str):
        """
        Initialize the class with the filename of the topological map.
        
        This class accept both a topological map and an orbit grid as input files
        
        

        """
        self.filename = filename
        self._readFile(filename)

    def _readFile(self, filename: str):
        logger.info('Reading the topological map from %s' % filename)
        dtree = xr.open_datatree(filename, engine='h5netcdf')
        if 'og' in dtree:
            logger.debug('File identified as orbit grid')
            raise NotImplementedError('Orbit grid reading not implemented yet')
        else:
            self.data = dtree.dataset
            # Set the coodinates
            self.data = self.data.swap_dims({
                'phony_dim_0':'E_array',
                'phony_dim_1':'R_array',
                'phony_dim_2':'p_array',
                'phony_dim_3':'z_array'
            }).rename({
                'E_array':'E0axis',
                'R_array':'R0axis',
                'p_array':'p0axis',
                'z_array':'z0axis'
            })
            self.data['R0axis'].attrs['long_name'] = 'R'
            self.data['R0axis'].attrs['units'] = 'm'
            self.data['z0axis'].attrs['long_name'] = 'z'
            self.data['z0axis'].attrs['units'] = 'm'
            self.data['E0axis'].attrs['long_name'] = 'E'
            self.data['E0axis'].attrs['units'] = 'eV'
            self.data['p0axis'].attrs['long_name'] = 'p'
            self.data['p0axis'].attrs['units'] = 'm'
            
            try:
                self.data['Wpol'] = 2.0*np.pi/self.data['polTransTimes']
                self.data['Wtor'] = 2.0*np.pi/self.data['torTransTimes']
            except KeyError:
                logger.warning('No transit times in file')

    def calculateResonances(self, omega: float, n: int, p: float,
                            threshold: float = 2.0, waiting_bar: bool=False):
        """
        Compute the resonance lines for a given set of (frequency, ntor, p).

        The resonant condition for trapped particles can be computed as:
            $$\omega_{mode} = n * \omega_{tor} + p * \omega_{bounce}$$

        where the particle frequencies have been computed by the fortran code
        and loaded previously. The resonant line(s) are computed by detecting
        the peaks in the following function:

            $$log(\Omega_{nr}/\omega_{mode}) > \text{threshold}$$

        being $\Omega_{nr}$ the departure from the resonance condition.

        Jose Rueda Rueda - jrrueda@us.es

        :param n: toroidal mode number(s). Must be integer values.
        :param p: transit/bounce harmonic number(s). Can be either integer,
        for linear resonance, or rational numbers, for non-linear resonances.
        :param threshold: consider a value to be resonant with given particle
        phase-space.
        :param waiting_bar: whether to display a waiting bar to track down the
        resonance calculation. Disabled by default.
        """
        # I have no time, I will improve this
        n = np.atleast_1d(n)
        p = np.atleast_1d(p)
        omega = np.atleast_1d(omega)
        # Generating the output: log of the (\Omega_{nl}/omega_res)
        res = np.zeros((n.size, p.size, omega.size,
                        *self.data.Wpol.shape), dtype=float)
        omega_res = np.zeros_like(self.data.Wpol.values)
        flagsNoOrbit = self.data.topoMap.astype(int).values == 9
        for ii, intor in tqdm(enumerate(n), disable=not waiting_bar):
            for jj, ipval in enumerate(p):
                omega_res = ipval * self.data.Wpol.values + \
                                           intor * self.data.Wtor.values
                # Computing for each of the frequencies how far
                # away are we from the resonance.
                for kk, iomega in enumerate(omega):
                    res[ii, jj, kk] = - np.log10(np.abs(omega_res-iomega) / iomega)

                    # Removing the lost orbits
                    res[ii, jj, kk, flagsNoOrbit] = np.nan
        logging.warning('Neglecting the part (n-mq) in the passing resonance')
        # Generating the corresponding xarray to contain the data properly labelled:
        coords = {'ntor': n, 'p': p, 'omega': omega}
        coords.update(self.data.Wpol.coords)
        res = xr.DataArray(res, dims=['ntor', 'p', 'omega', *self.data.Wpol.dims],
                            coords=coords,
                            attrs={ 'long_name': 'Resonance condition',
                                    'units': 'log10',
                                    'name': 'Resonance'})

        # Setting the qualifiers of the grid basis.
        res['ntor'].attrs['long_name'] = '$n_{tor}$'
        res['ntor'].attrs['units'] = ''

        res['p'].attrs['long_name'] = 'p'
        res['p'].attrs['units'] = ''

        res['omega'].attrs['long_name'] = r'$\omega$'
        res['omega'].attrs['units'] = 'Hz'

        # for icoord in self.data.Wpol.dims:
        #     res[icoord].attrs['long_name'] = self.data[icoord].long_name
        #     res[icoord].attrs['units'] = self.data[icoord].units

        return resonance_viewer(res, omega_tor=self.data.Wtor,
                                omega_pol=self.data.Wpol)
        

    def plot(self, var: str = 'Wtor', overlayRegions = True, ax=None,
             R0=None, z0=None, E0=None, p0=None):
        """
        Plot the data as a 2D colormapped plot.

        Jose Rueda Rueda - jrrueda@us.es
        Pablo Oyola - poyola@us.es

        :param var: variable to plot, defaults to 'Wtor'.
        :param overlayRegions: if True, the topological regions are overlayed.
        :param ax: axis to plot, if None, a new figure is created.
        """
        # ---- Check input
        if var not in self.data:
            raise ValueError(f'The input {var} is not stored in the data.')

        count = 0
        c = {'R0axis': R0, 'z0axis': z0, 'E0axis': E0, 'p0axis': p0}
        d = c.copy()
        for k in c.keys():
            if c[k] is None:
                d.pop(k)
                count += 1

        if count != 2:
            raise Exception('Only 2 variables can be specified!')

        # --- Create axis if needed
        if ax is None:
            fig, ax = plt.subplots()

        # ---- Plot
        im = self.data[var].sel(method='nearest', **d).plot.imshow(robust=False)
        if overlayRegions:
            self.data['Kind'].sel(method='nearest', **d).plot.imshow(
                cmap='Pastel1', alpha=0.25, add_colorbar=False)

        return im


    def plot_gui(self, *args, var: str = 'Wtor', overlayRegions = True,
                 **kwargs):
        """
        Set a GUI for plotting and varying the phase-space representation.

        Pablo Oyola - poyola@us.es

        :param args: variable name to use for the sliders.
        :param var: variable to plot.
        :param overlayRegions: if True, the regions are overlayed in the plot.
        :param kwargs: additional arguments to be passed to the plot function.
        :return: axis and sliders objects.
        """
        # Check input
        if var not in self.data:
            raise ValueError(f'The input {var} is not stored in the data.')

        # Let's check which variables to use for the plot and which ones to
        # use for the bars.
        vars_plot = list(self.data[var].coords.keys())
        for iargs in args:
            if iargs+'0axis' not in self.data:
                raise ValueError('Variable %s not found in the dataset' % (iargs+'0axis'))

            for ii, icoords in enumerate(vars_plot):
                if icoords.lower() == (iargs.lower()+'0axis'):
                    vars_plot.pop(ii)

        # Removing the ID coordinate.
        vars_plot = [ii for ii in vars_plot if (ii.lower() != 'id') and \
                                               (self.data[ii].size > 1)]

        # We will now check how many slider-variables are actually
        # varying.
        slider_vars = list()
        fix_vars    = list()
        for iargs in args:
            if self.data[iargs+'0axis'].size > 1:
                slider_vars.append(iargs+'0axis')
            else:
                fix_vars.append(iargs+'0axis')

        # Creating the figure layout.
        fig, ax = plt.subplots(1)

        # Base updater.
        def base_updater(slider_var: str, val):
            # Updating the plotter.
            slider_vars_val[slider_var] = val
            data2plot = self.data[var].sel(method='nearest',
                                           **slider_vars_val).squeeze()
            data2plot = data2plot.transpose(*vars_plot).values
            im.set_data(data2plot.T)

            # We also update the corresponding colorbar
            if var.lower() != 'kind':
                im.set_clim(vmin=data2plot.min(), vmax=data2plot.max())

        # Creating the axes divider.
        ax_div = make_axes_locatable(ax)
        axes_sliders = list()
        slider_vars_val = dict()
        sliders = list()
        copy_var = slider_vars.copy()
        for ii, ivar in enumerate(slider_vars):
            iax = ax_div.append_axes('bottom', '5%', pad='15%')
            axes_sliders.append(iax)
            islider = Slider(
                ax=iax,
                label='%s (%s)' % (self.data[ivar].long_name,
                                   self.data[ivar].units),
                valstep=self.data[ivar].values,
                valinit=self.data[ivar].values[0],
                valmin=self.data[ivar].values.min(),
                valmax=self.data[ivar].values.max()
            )
            islider.on_changed(lambda val: base_updater(copy_var[ii], val))

            slider_vars_val[ivar] = self.data[ivar].values[0]
            sliders.append(islider)

        for ii, ivar in enumerate(fix_vars):
            slider_vars_val[ivar] = self.data[ivar].values[0]

        extent = [(self.data[ivar].min(), self.data[ivar].max()) \
                  for ivar in vars_plot]

        extent = np.array(extent).flatten()

        if var.lower() == 'kind':
            if 'cmap' in kwargs:
                kwargs.pop('cmap')
            if 'vmin' in kwargs:
                kwargs.pop('vmin')
            if 'vmax' in kwargs:
                kwargs.pop('vmax')

            kwargs['cmap'] = plt.matplotlib.colors.ListedColormap(['black',
                                                                    'red',
                                                                    'blue',
                                                                    'green',
                                                                    'cyan',
                                                                    'magenta'])
            kwargs['vmin'] = 0
            kwargs['vmax'] = 6

        data2plot = self.data[var].sel(method='nearest', **slider_vars_val).squeeze()
        data2plot = data2plot.transpose(*vars_plot).values
        im = ax.imshow(data2plot.T, extent=extent, origin='lower', aspect='auto',
                       **kwargs)

        # Setting up the plot decorations.
        labels = ['%s (%s)' % (self.data[ii].long_name, self.data[ii].units)
                  for ii in vars_plot]

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        cbar = ax_div.append_axes('right', '5%', pad='7%')

        # If the variable to be plot is the Kind, then we use a segmented
        # colorbar.
        cbarobj = fig.colorbar(mappable=im, cax=cbar)
        if var.lower() == 'kind':
            ticklabels = ['?', 'Loss', 'Trapped',
                          'Stagnated', 'Co', 'Cntr']
            cbarobj.set_ticks(np.arange(6) + 0.5,
                              labels=ticklabels)

        return ax, im, sliders


class resonance_viewer:
    """
    Store the resulting histogram from the marker_diag class.
    Allow the user to simply plot the results and to save them in a file.
    """
    def __init__(self, data: Union[xr.DataArray, str, Iterable],
                 omega_tor: xr.DataArray = None, omega_pol: xr.DataArray = None,):
        """
        Initialize the class with the resulting data.

        Pablo Oyola - poyola@us.es

        :param data: Either the path to the file containing the netcdf data,
        the xarray itself (resulting from the corresponding call), or a list
        (path, variable) in case the data is stored into a netcdf file with
        more variables.
        """
        if isinstance(data, str):
            # This assumes there is only field stored in the file.
            self.data = xr.open_dataset(data)

            # Transform dataset into a dataarray.
            self.data = self.data[list(self.data.data_vars.keys())[0]]
        elif isinstance(data, xr.DataArray):
            self.data = data
        elif isinstance(data, Iterable):
            # The first element isf the file.
            if not isinstance(data[0], str):
                raise TypeError('The first element of the iterable must be a string.')

            # Checking if the file exists.
            if not os.path.isfile(data[0]):
                raise FileNotFoundError(f'The file provided does not exist: {data[0]}')

            # Checking if the variable is inside the file.
            with nc.Dataset(data[0], 'r') as f:
                if data[1] not in f.variables.keys():
                    raise ValueError(f'The variable {data[1]} is not in the file {data[0]}.')

                # We read the data.
                tmp = np.array(f.variables[data[1]][:], dtype=f.variables.dtype)

                # We also read the corresponding coordinates.
                coords = {}
                for ikey in f.variables[data[1]].dimensions:
                    coords[ikey] = np.array(f.variables[ikey][:], dtype=f.variables[ikey].dtype)

                # We create the DataArray.
                self.data = xr.DataArray(tmp, dims=f.variables[data[1]].dimensions,
                                         coords=coords, attrs=f.variables[data[1]].__dict__)
        else:
            raise TypeError('The data provided is not valid.')

        # We add the parameter name to the data array.
        if self.data.name is None:
            self.data.name = self.data.long_name

        self.omega_tor = omega_tor
        self.omega_pol = omega_pol
        self._prepare_p_matrix()

    def _prepare_p_matrix(self):
        """
        Prepare the p-matrix used to locate in the phase-space the resonances.

        The p-matrix is defined on the phase-space of the particles. It is
        computed as follows:

            \\omega - n\\omega_\\phi
        p = ---------------------
                \\omega_\\theta

        Pablo Oyola - poyola@us.es
        Jose Rueda Rueda - jrrueda@us.es
        """
        if self.omega_tor is None:
            logger.warning('omega_tor not provided. Cannot compute p-matrix')
            self.pmatrix = None
            return

        if self.omega_pol is None:
            logger.warning('omega_pol not provided. Cannot compute p-matrix')
            self.pmatrix = None
            return

        self.pmatrix = (self.data.omega.values - self.data.ntor.values * self.omega_tor) / self.omega_pol

        self.pmatrix.attrs['long_name'] = 'p-matrix'
        self.pmatrix.attrs['units'] = '-'
        self.pmatrix.attrs['description'] = 'p-matrix used to locate the resonances in the phase-space.'
        self.pmatrix.attrs['name'] = 'p-matrix'

    def gui(self, *names, stacked_var: str=None, exp: bool=False, **kwargs):
        """
        Generate a GUI to plot the data.

        Pablo Oyola - poyola@us.es

        :param names: name(s) to use in the plot. The rest will be set
        as sliders.
        :param stacked_var: variable to be stacked in the plot in case
        of the variable being 1D.
        :param exp: if True, the data will be plotted in linear scale. Otherwise,
        it will be plotted in logarithmic scale.
        :param kwargs: additional arguments to be passed to the plot.
        """
        # Checking the names are in the data coordinates.
        for name in names:
            if not name in self.data.coords.keys():
                raise ValueError('%s is not in the data coordinates.'%name)

        # Lets' check if the stacked_vars is in the data coordinates.
        if stacked_var is not None:
            if len(names) != 1:
                raise ValueError('Cannot do stacked plot with more than one variable.')
            if stacked_var not in self.data.coords.keys():
                raise ValueError('The stacked_vars is not in the data coordinates.')


        # Checking which variables are just single-valued.
        single_valued = []
        for ikey in self.data.coords.keys():
            if len(self.data.coords[ikey]) == 1:
                single_valued.append(ikey)

        # We remove the single-valued variables from the names.
        names = [iname for iname in names if iname not in single_valued]

        # We set the rest of the variables of the data as sliders coordinates.
        sliders_vars = [iname for iname in self.data.coords.keys()
                        if (iname not in names) and (iname not in single_valued) and \
                           (iname != stacked_var)]

        # Defining auxiliary functions to update the plots.
        def update_slider_1d(slider_var: str, val):
            # Updating the plotter.s
            slider_vars_val[slider_var] = val
            data2plot = self.data.sel(method='nearest',
                                      **slider_vars_val).squeeze()

            # If we have exponential flag set, then we plot in linear scale.
            if exp:
                data2plot = np.exp(data2plot)

            # The data may now be 2D, in case a stacked plot is requested.
            if stacked_var is not None:
                for ii in range(data2plot[stacked_var].size):
                    im[ii].set_ydata(data2plot.isel({stacked_var: ii}).values)
            else:
                im.set_ydata(data2plot)

        def update_slider_2d(islider_var_name: str, val):
            # Updating the plotter.
            slider_vars_val[islider_var_name] = val
            data2plot = self.data.sel(method='nearest',
                                      **slider_vars_val).squeeze()

            # If we have exponential flag set, then we plot in linear scale.
            if exp:
                data2plot = np.exp(data2plot)

            # Ordering the data.
            # data2plot = data2plot.squeeze().transpose(*names)

            # Printing for debug the slider vars.
            # im.set_data(data2plot.values)
            im = data2plot.plot.imshow(ax=ax, origin='lower',
                                      add_colorbar=False, **kwargs)
            # Update the colorbar limits.
            vmin = np.nanmin(data2plot.values[np.isfinite(data2plot)])
            vmax = np.nanmax(data2plot.values[np.isfinite(data2plot)])
            im.set_clim(vmin=vmin, vmax=vmax)
            cbarobj = fig.colorbar(mappable=im, cax=cbar)
            cbarobj.set_label(f'{self.data.long_name} ({self.data.units})')
            fig.canvas.draw_idle()

        # Opening a new window.
        fig, ax = plt.subplots(1)

        ax_div = make_axes_locatable(ax)

        # Setting the sliders.
        sliders = {}
        for ikey in sliders_vars:
            # Appending axes to the figure for the sliders.
            iax = ax_div.append_axes('bottom', '5%', pad='15%')

            # We set the limits of the slider.
            vmin = np.nanmin(self.data.coords[ikey])
            vmax = np.nanmax(self.data.coords[ikey])

            # Generating the description of the sliders.
            idesc = f'{self.data.coords[ikey].long_name} ({self.data.coords[ikey].units})'

            # We create the slider.
            sliders[ikey] = Slider(valinit=vmin, valmin=vmin,
                                   valmax=vmax,  label=idesc,
                                   valstep=self.data.coords[ikey].values,
                                   ax=iax)

        # Allocating temporary variables to store the values of the sliders.
        slider_vars_val = {}
        for ivar in sliders_vars:
            slider_vars_val[ivar] = self.data[ivar].values[0]

        # Distinguishing between 1D or 2D plots.
        if len(names) == 1:
            # If the user requested a stacked plot, we do a loop.
            data2plot = self.data.sel(method='nearest',  **slider_vars_val)
            if exp:
                data2plot = np.exp(data2plot)
            if stacked_var is not None:
                im = list()
                for ii in range(self.data[stacked_var].size):

                    tmp, = ax.plot(self.data[names[0]].values,
                                   data2plot.isel({stacked_var: ii}).values.squeeze(),
                                   label=f'{stacked_var} = {self.data[stacked_var].values[ii]}')
                    im.append(tmp)
            else:
                im, = ax.plot(self.data[names[0]].values, data2plot.values.squeeze())

            # Setting up the plot decorations.
            xlabel = self.data[names[0]].long_name + ' (' + self.data[names[0]].units + ')'
            ylabel = self.data.long_name + ' (' + self.data.units + ')'

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            ax.grid('both')

            # Setting up the sliders updating function.
            funcs = {ikey: lambda val, ikey0=ikey: update_slider_1d(deepcopy(ikey0),
                                                                    val) \
                     for ikey in sliders_vars}
            for ikey in sliders_vars:
                sliders[ikey].on_changed(funcs[ikey])

        elif len(names) == 2:
            # Setting the extent for imshow 2D plot.
            extent = [(self.data[ivar].min(), self.data[ivar].max()) \
                       for ivar in names]

            extent = np.array(extent).flatten()

            data2plot = self.data.sel(method='nearest', **slider_vars_val).squeeze()
            if exp:
                data2plot = np.exp(data2plot)

            # data2plot = data2plot.transpose(*names)

            # im = ax.imshow(data2plot.T, extent=extent, origin='lower',
            #                aspect='auto', **kwargs)
            im = data2plot.plot.imshow(ax=ax, origin='lower',
                                       add_colorbar=False, **kwargs)
            # Setting up the plot decorations.
            labels = ['%s (%s)' % (self.data[ii].long_name, self.data[ii].units)
                       for ii in names]

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            cbar = ax_div.append_axes('right', '5%', pad='7%')

            # If the variable to be plot is the Kind, then we use a segmented
            # colorbar.
            cbarobj = fig.colorbar(mappable=im, cax=cbar)
            cbarobj.set_label(f'{self.data.long_name} ({self.data.units})')

            # Setting up the slider updating functions.
            funcs = {ikey: lambda val, ikey0=ikey: update_slider_2d(deepcopy(ikey0), val) for ikey in sliders_vars}
            for ikey in sliders_vars:
                sliders[ikey].on_changed(funcs[ikey])
        else:
            raise NotImplementedError('Only 1D and 2D plots are implemented.')


        # Return the axes and sliders.
        return ax, im, sliders, funcs

    def rebinning(self, method: str='MC', **bins):
        """
        Rebin the data into a new grid.

        Use this for lowering the resolution. Uppening the resolution
        only leads to aliasing.

        Pablo Oyola - poyola@us.es

        :param method: method to use for rebinning. Options are:
                          - MC: Monte Carlo rebinning.
                          - NN: Nearest neighbour rebinning.

        :param bins: dictionary with {name : bin_number} pairs.
        """
        # Checking all the names in bins exists as coords of the
        # data.
        for iname in bins:
            if iname not in self.data.coords:
                raise ValueError(f'{iname} is not a coordinate of the data.')

        # Creating the new grids.
        new_coords = {iname: np.linspace(self.data.coords[iname].min(),
                                            self.data.coords[iname].max(),
                                            bins[iname]) \
                        for iname in bins}


        if method.lower() == 'mc':
            # Generating a list with the bins.
            bins_list = [bins[iname] if iname in bins \
                         else self.data[iname].size \
                         for iname in self.data.dims]
            dx = [self.data.coords[iname].values[1] - self.data.coords[iname].values[0] \
                    for iname in self.data.dims]
            tmp = fast_mc_rebinning(self.data.values.squeeze(),
                                    bins_list, dx, nrandom=10)
            data_out = xr.DataArray(tmp, dims=self.data.dims, coords=new_coords,
                                    attrs=self.data.attrs)

            # We also need to update the coordinates.
            for iname in data_out.dims:
                data_out[iname].attrs = self.data[iname].attrs
        elif method.lower() == 'nn':
            # We use a nearest neighbour rebinning: aka, linear interpolation.
            data_out = self.data.interp(**new_coords)
        else:
            raise ValueError(f'{method} is not a valid rebinning method.')

        # Generating another diagnostic_viewer instance.
        return resonance_viewer(data_out)

    def to_netcdf(self, file: str, mode: str='w', compress: bool=True):
        """
        Save the data in a NetCDF file.

        Pablo Oyola - poyola@us.es

        :param file: The file to save the data.
        :param mode: The mode to open the file. There are several options.
            - 'w': Create a new, empty file.
            - 'a': Append data to an existing file.
        :param compress: If True, the data will be compressed.
        """
        if mode not in ['w', 'a']:
            raise ValueError(f'{mode} is not a valid mode.')

        encoding = {}
        # if compress:
        #     encoding = {iname: {'zlib': True, 'complevel': 9} \
        #                 for iname in self.data}
        self.data.to_netcdf(file, mode=mode, encoding=encoding)

    def project(self, p: float, neprime: int=100, nmu: int=101,
                nEbase_res: int=1024, Ebase_min: float=5.0,
                waiting_bar: bool=False):
        """
        Project along the (E', mu) the resonance condition, allowing to study
        the transport.

        Pablo Oyola - poyola@us.es
        """
        # We will work for now with only one frequency.
        if self.data.omega.size > 1:
            raise NotImplementedError('Only one frequency is implemented.')

        if self.data.ntor.size > 1:
            raise NotImplementedError('Only one ntor is implemented.')

        # We generate the grid in mu and E'.
        mu = np.linspace(self.data.mu0axis.values.min(),
                         self.data.mu0axis.values.max(),
                         nmu)

        EE, PP = np.meshgrid(self.data.E0axis.values,
                             self.data.pphi0axis.values)
        Eprime = EE - self.data.omega.values / self.data.ntor.values * PP / ec * 1.0e-3

        Eprimegrid = np.linspace(Eprime.min(), Eprime.max(), neprime)

        # We generate the grid in E.
        Ebase = np.linspace(self.data.E0axis.values.min(),
                            self.data.E0axis.values.max(),
                            nEbase_res)

        # Loop over Eprimegrid and mu.
        data_out = np.zeros((neprime, nmu))
        y_envelope = np.zeros((neprime,)) + np.nan
        x_envelope = np.zeros((neprime,))
        data = self.data.sel(p=p).isel(ntor=0, omega=0)
        flags_energy = Ebase < Ebase_min

        basis4interpol = (self.data.E0axis,
                          self.data.mu0axis,
                          self.data.pphi0axis)

        for iEprime, iEprime0 in enumerate(tqdm(Eprimegrid, disable=not waiting_bar)):
            for imu, imu0 in enumerate(mu):
                # We compute the pphi for the given Eprime and for
                # all the Ebase.
                pphibase = self.data.ntor.values / self.data.omega.values * \
                            (Ebase - iEprime0) * ec * 1.0e3
                mubase = np.ones_like(pphibase) * imu0

                # We interpolate the data.
                tmp = RegularGridInterpolator(basis4interpol,
                                              data.values.squeeze(),
                                              bounds_error=False)((Ebase, mubase, pphibase))


                tmp[flags_energy] = np.nan

                # Finding the maxima
                try:
                    idxmax = np.nanargmax(tmp)
                except ValueError:
                    data_out[iEprime, imu] = np.nan
                    continue

                Emax = Ebase[idxmax]
                flags = np.logical_not(np.isnan(tmp)) & (Ebase < 0.99 * Emax)

                idxs = np.where((tmp[flags] - 1) < 1e-2)
                try:
                    val = Ebase[flags][idxs[0][-1]]
                except:
                    val = np.nan

                data_out[iEprime, imu] = val - Emax

            # We finished finding the curve for a given Eprime.
            # Let's get the mu for which we have a maximum
            if np.all(np.isnan(data_out[iEprime, :])):
                continue

            idxmax = np.nanargmin(data_out[iEprime, :])
            x_envelope[iEprime] = mu[idxmax]
            y_envelope[iEprime] = data_out[iEprime, idxmax]

        # We generate the output.
        data_out = xr.DataArray(data_out, dims=['Eprime', 'mu'],
                                coords={'Eprime': Eprimegrid,
                                        'mu': mu},
                                attrs={'long_name': 'Resonance width',
                                       'units': 'keV',
                                       'p': p,})
        data_out.Eprime.attrs = {'long_name': 'Eprime',
                                 'name': 'E\'',
                                 'units': 'keV'}
        data_out.mu.attrs = {'long_name': 'mu',
                             'name': '$\\mu$',
                             'units': 'keV/T'}

        # We interpolate the envelope on the mu.
        flags = np.logical_not(np.isnan(y_envelope))
        f = np.interp(mu, x_envelope[flags], y_envelope[flags])

        output = xr.Dataset()
        output['reswidth'] = data_out
        output['dEmax'] = xr.DataArray(f, dims=['mu'],
                                        coords={'mu': mu},
                                        attrs={'long_name': 'Maximum resonance width',
                                                 'units': 'keV'})

        return output
