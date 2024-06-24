"""
Methods to determine the sensitivity of the tomography algorithm to noise 
and artefacts.
"""
import copy
import os
import ScintSuite as ss
import numpy as np
import tkinter as tk
import xarray as xr
import ScintSuite._Tomography._synthetic_signal as synthetic_signal


def noise_sensitivity(WF, inverter, window, iters, noise_levels, 
                      max_noise = 0.15):
        '''
        This function calculates the fidelity map of the tomography algorithm.
        For each noise level, the function generates a synthetic signal, adds
        noise to it and performs the tomography with the algebraic algorithm 
        selected and the number of iterations selected. The function returns 
        the fidelity map asociated. Each element of the fidelity map is the
        Signal to Noise Ratio (SNR) of the reconstruction of a delta placed in 
        that pixel.

        Marina Jimenez

        Parameters:
        -----------
        WF: xarray.DataArray
            Weight function.
        inverter: str
            Algebraic algorithm to be used. Pick just one algebraic algorithm:
            'descent', 'kaczmarz' or 'cimmino'
        window: list
            Window of the grid to be used for the map.
        iters: int
            Maximum number of iterations.

        max_noise: float
            Maximum noise level that the reconstructed image is allowed to have.
              The default is 0.15.

        '''
        # Generate grid
        # gyroradius for x and pitch for y
        r_values = WF.y.values
        p_values = WF.x.values

        r_liminf = r_values [r_values >= window[2]]
        p_liminf = p_values[p_values >= window[0]]

        r_selected = r_liminf[r_liminf <= window[3]]
        p_selected = p_liminf[p_liminf <= window[1]]

        noise_sensitivityXR = xr.DataArray(np.nan*np.empty((len(p_selected), 
                                                       len(r_selected))), 
                                coords=[('x', p_selected), ('y', r_selected)])
        
        # Loop over the index of r_values
        for i in np.arange(0,len(r_selected)):            
            # Loop over the index of p_values
            for j in np.arange(0,len(p_selected)):
                # Access the j-th element of p_values
                
                mu_gyro = r_selected[i]
                mu_pitch = p_selected[j]
                noise_sensitivityXR[j,i] = noise_levels.max()

                for n in noise_levels:
                    # Generate the synthetic signal
                    x, y = synthetic_signal.create_synthetic_delta(WF, mu_gyro, 
                                                        mu_pitch,
                                                        noise_level = n,
                                                        background_level = 0.01,
                                                        seed=0)
                    if y.max() == 0:
                        noise_sensitivityXR[j,i] = 0
                        continue

                    # Perform the tomography
                    tomo = ss.tomography(WF, y)
                    x0 = np.zeros(tomo.s1D.shape)
                    if inverter == 'descent':
                        tomo.coordinate_descent_solve(x0, iters,  damp = 0.1, 
                                                relaxParam = 1)
                    elif inverter == 'kaczmarz':
                        tomo.kaczmarz_solve(x0, iters, damp = 0.1, 
                                            relaxParam = 1)
                    elif inverter == 'cimmino':
                        tomo.cimmino_solve(x0, iters, damp = 0.1, 
                                           relaxParam = 1)
                    
                    xHat = tomo.inversion[inverter].F.isel(alpha = 0).copy()
                    MSE = np.sqrt(((xHat-x)**2).sum(dim=('x','y')))
                    true_norm = np.sqrt((x**2).sum(dim=('x','y')))
                    error = 1 - MSE/true_norm
                    if error <= max_noise:
                        noise_sensitivityXR[j,i] = n

                    

        return noise_sensitivityXR
                        



def fidelity_map(WF, inverter, window, iters, noise):
        '''
        This function calculates the fidelity map of the tomography algorithm.
        For each noise level, the function generates a synthetic signal, adds
        noise to it and performs the tomography with the algebraic algorithm 
        selected and the number of iterations selected. The function returns 
        the fidelity map asociated. Each element of the fidelity map is the
        Signal to Noise Ratio (SNR) of the reconstruction of a delta placed in 
        that pixel.

        Marina Jimenez

        Parameters:
        -----------
        WF: xarray.DataArray
            Weight function.
        inverter: str
            Algebraic algorithm to be used. Pick just one algebraic algorithm:
            'descent', 'kaczmarz' or 'cimmino'
        window: list
            Window of the grid to be used for the map.
        iters: int
            Maximum number of iterations.
        noise: float
            Noise level.

        '''
        # Generate grid
        # gyroradius for x and pitch for y
        r_values = WF.y.values
        p_values = WF.x.values

        r_liminf = r_values [r_values >= window[2]]
        p_liminf = p_values[p_values >= window[0]]

        r_selected = r_liminf[r_liminf <= window[3]]
        p_selected = p_liminf[p_liminf <= window[1]]

        fidelity_mapXR = xr.DataArray(np.nan*np.empty((len(p_selected), 
                                                       len(r_selected))), 
                                coords=[('x', p_selected), ('y', r_selected)])
        
        # Loop over the index of r_values
        for i in np.arange(0,len(r_selected)):            
            # Loop over the index of p_values
            for j in np.arange(0,len(p_selected)):
                # Access the j-th element of p_values
                
                mu_gyro = r_selected[i]
                mu_pitch = p_selected[j]

                    # Generate the synthetic signal
                x, y = synthetic_signal.create_synthetic_delta(WF, mu_gyro, 
                                                        mu_pitch,
                                                        noise_level = noise,
                                                        background_level = 0.01,
                                                        seed=0)
                if y.max() == 0:
                    fidelity_mapXR[j,i] = 0
                    continue

                # Perform the tomography
                tomo = ss.tomography(WF, y)
                x0 = np.zeros(tomo.s1D.shape)
                if inverter == 'descent':
                    tomo.coordinate_descent_solve(x0, iters,  damp = 0.1, 
                                                relaxParam = 1)
                elif inverter == 'kaczmarz':
                    tomo.kaczmarz_solve(x0, iters, damp = 0.1, 
                                            relaxParam = 1)
                elif inverter == 'cimmino':
                    tomo.cimmino_solve(x0, iters, damp = 0.1, 
                                           relaxParam = 1)
                    
                xHat = tomo.inversion[inverter].F.isel(alpha = 0).copy()
                MSE = np.sqrt(((xHat-x)**2).sum(dim=('x','y')))
                true_norm = np.sqrt((x**2).sum(dim=('x','y')))
                SNR = 20*np.log10(true_norm/MSE)
                fidelity_mapXR[j,i] = SNR
                    

        return fidelity_mapXR

