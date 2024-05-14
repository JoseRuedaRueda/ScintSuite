"""
Method to create a synthetic signal for testing purposes

Marina Jimenez Comez - mjimenez37@us.es
"""

import numpy as np
import xarray as xr
import ScintSuite._Tomography._martix_collapse as matrix
from scipy.stats import multivariate_normal

def create_synthetic_signal(WF, mu_gyro, mu_pitch, power, sigma_gyro,  
                            sigma_pitch, noise_level, background_level,
                              seed = 0):
    """
    Generate a synthetic signal in the pinhole for testing purposes
    and a synthetic frame assosiated.

    Parameters
    ----------
    WF : xarray.DataArray
        Weight function.
    mu_gyro : list
        Mean value of the gyro angle.
    power : list
        Power of the signal.
    sigma_gyro : float
        Standard deviation of the gyro angle.
    mu_pitch : list
        Mean value of the pitch angle.
    sigma_pitch : float
        Standard deviation of the pitch angle.
    noise_level : float
        Noise level.
    background_level : float
        Background level.
    seed : int, optional
        Seed for the random number generator. The default is 0.

    Returns
    -------
    frame_synthetic : xarray.DataArray
        Synthetic frame.

    :Example:
    --------
    WF = xr.load_dataarray(WFfile)
    mu_gyro = [3.1, 4.3, 5.4] 
    power = [0.1, 0.2, 0.7] 
    sigma_gyro = 0.01
    mu_pitch = [55, 55, 55]
    sigma_pitch = 7
    noise_level = 0.1
    background_level = 0.1
    x, y = synthetic_signal.create_synthetic_signal(WF, mu_gyro, mu_pitch,
                                                    power, sigma_gyro, 
                                                    sigma_pitch, noise_level,
                                                    background_level,
                                                    seed=seed)


    """
    # Generate grid
    gyro_grid = WF.y.values
    pitch_grid = WF.x.values
    X, Y = np.meshgrid(WF.y, WF.x) # gyroradius for x and pitch for y
    grid = np.dstack((X, Y))

    # Generate synthetic signal
    x_synthetic = np.zeros_like(X)

    for i in range(len(mu_gyro)):
        distrib = multivariate_normal([mu_gyro[i], mu_pitch[i]],
                                            [[sigma_gyro, 0],
                                            [0, sigma_pitch]], seed=seed)
        pdf = distrib.pdf(grid)
        x_synthetic += pdf/pdf.max()*power[i]

    x_synthetic[x_synthetic < 0.001*x_synthetic.max()]=0


    # Final synthetic signal
    x_syntheticXR = xr.DataArray(data=x_synthetic,
                                    dims=['x', 'y'],
                                    coords=dict(
                                        x = (['x'], pitch_grid),
                                        y = (['y'], gyro_grid))
                                    )
    
    WF_2D = matrix.collapse_array4D(WF.values)
    x_synthetic1D = matrix.collapse_array2D(x_synthetic)
    y_synthetic1D = WF_2D @ x_synthetic1D

    # Add noise
    noise1 = noise_level * y_synthetic1D * np.random.randn(len(y_synthetic1D))
    max_value = y_synthetic1D.max()
    noise2 = background_level * max_value * np.random.randn(len(y_synthetic1D))
    combined_noise = np.maximum(noise1, noise2)
    y_synthetic1D = y_synthetic1D + combined_noise
    y_synthetic1D = y_synthetic1D / y_synthetic1D.max()

    # Final synthetic frame
    nx = x_synthetic.shape[0]
    ny = x_synthetic.shape[1]
    y_synthetic = matrix.restore_array2D(y_synthetic1D, nx, ny)

    frame_synthetic = xr.DataArray(data=y_synthetic,
                                      dims=['x', 'y'],
                                      coords=dict(
                                        x = (['x'], pitch_grid),
                                        y = (['y'], gyro_grid)))
    
    return x_syntheticXR, frame_synthetic
