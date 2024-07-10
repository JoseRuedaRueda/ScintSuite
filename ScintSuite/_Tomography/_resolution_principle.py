"""
Methods to implement the resolution principle in the tomography module.
"""
import copy
import os
import ScintSuite as ss
import numpy as np
import xarray as xr
from scipy.ndimage import label
import ScintSuite._Tomography._synthetic_signal as synthetic_signal


def binarize_xarray(xarray, threshold):
    '''
    Binarize an xarray DataArray.

    Parameters
    ----------
    xarray : xarray.DataArray
        The xarray DataArray to binarize.
    threshold : float
        The threshold to use for the binarization.

    Returns
    -------
    xarray.DataArray
        The binarized xarray DataArray.
    '''
    return (xarray > threshold).astype(int)


def calculate_centroid(labeled, ncomponents, xHat):
    '''
    Calculate the centroid of an xarray DataArray.

    Parameters
    ----------
    labeled : np.array
        The labeled image.
    ncomponents : int
        The number of components in the image.
    xHat : xarray.DataArray
        The xarray DataArray to calculate the centroid of.

    Returns
    -------
    np.array
        The centroid of the xarray DataArray.
    '''
    ar = xHat.values
    
    labeledXR = xr.DataArray(labeled, dims=xHat.dims, coords=xHat.coords)

    centroids = []
    max_intensity = []

    # Label 0 is considered as background. Therefore, the
    # total number of labels in the image are the number of components + 1
    for l in np.arange(1, ncomponents+1):
        # Extract coords of each label
        coords = labeledXR.where(labeledXR == l, drop=True).coords
        values = xHat.sel(x = coords['x'], y = coords['y'])
        centr_x = float(np.sum(values.sum('y')/np.sum(values.sum('y')) * values.x))
        centr_y = float(np.sum(values.sum('x')/np.sum(values.sum('x')) * values.y))
        centroid = [centr_x, centr_y]
        centroids.append(centroid)
        max_intensity.append(values.max())
    
    return centroids, max_intensity


def threshold_centroids(centroids, max_intensity, threshold):
    '''
    Take centroids with maximum intensity above a threshold.

    Marina Jimenez Comez: mjimenez37@us.es

    Parameters
    ----------
    centroids : np.array
        Centroids of the components.
    max_intensity : np.array
        Maximum intensity of the components.
    threshold : float
        Threshold to filter the centroids.

    Returns
    -------
    np.array
        Centroids of the components.

    '''
    centroids = np.array(centroids)
    max_intensity = np.array(max_intensity)

    # Filter centroids
    centroids = centroids[max_intensity > threshold]
    max_intensity = max_intensity[max_intensity > threshold]

    # Order centroids by intensity
    order = np.argsort(max_intensity)
    centroids = centroids[order]
    max_intensity = max_intensity[order]

    # Take the first 2 centroids
    peaks = centroids[:2]
    # intensity = max_intensity[:2]
    
    return peaks

def check_separation(peaks, xHat, map_type='pitch'):
    '''
    Check separation between peaks.

    Marina Jimenez Comez:

    Parameters
    ----------
    peaks : np.array
        Peaks to check.
    xHat : xarray.DataArray
        Tomography solution.
    map_type : str
        Type of map to check. Options are 'pitch' and 'gyro'.

    Returns
    -------
    bool
        True if the peaks are separated, False otherwise.

    '''

    # Grid resolution in pitch and gyro
    gridPitch = np.diff(xHat.x)[0]
    gridGyro = np.diff(xHat.y)[0]

    # Calculate the separation in pitch
    separationPitch = np.abs(np.diff(peaks[:,0], axis=0))
    
    # Calculate the separation in gyro
    separationGyro = np.abs(np.diff(peaks[:,1], axis=0))
    

    if map_type == 'pitch':
        if (separationPitch >= gridPitch) and (separationGyro <= 3*gridGyro):
            separated = True
            return True

    elif map_type == 'gyro':
        if (separationGyro >= gridGyro) and (separationPitch <= 6*gridPitch):
            return True
        

    
    return False

def get_peaks_connected_components(xHat, peak_amplitude=0.15):

    '''
    Get the peaks of the tomography solution xHat. The method is based on 
    detecting the connected components of the binary image of the tomography.

    Marina Jimenez Comez:

    Parameters
    ----------
    xHat : xarray.DataArray
        Tomography solution.
    peak_amplitude : float
        Peak amplitude threshold.

    Returns
    -------
    peaks : np.array
        Peaks of the tomography solution.

    '''

    # Transform the xHat to a binary image
    thresh = xHat.max() * 0.1
    xHat_binary = binarize_xarray(xHat, threshold=thresh)

    # Connected components: 8-connectivity
    structure = np.ones((3, 3), dtype=int)
    labeled, ncomponents = label(xHat_binary, structure)

    # Calculate centroids and max intensity of the components
    centroids, max_intensity = calculate_centroid(labeled, ncomponents, xHat)

    # Take centroids with maximum intensity above a threshold
    peaks = threshold_centroids(centroids, max_intensity, 
                                                   threshold=peak_amplitude)

    return peaks



def calculate_resolution_point(xHat, original_distance, map_type='pitch'):
    '''
    Calculate the resolution map for an specific point
      from the tomography solution xHat.

    Marina Jimenez Comez: mjimenez37@us.es

    Parameters
    ----------
    xHat : xarray.DataArray
        Tomography solution.
    original_distance : float
        Original distance between the peaks.
    map_type : str
        Type of resolution map to calculate. Options are 'pitch' and 'gyro'.

    '''
    # Get peaks
    peaks = get_peaks_connected_components(xHat, peak_amplitude=0.15)

    # Check separation between peaks
    separated = check_separation(peaks, xHat, map_type=map_type)

    resolution = 0.0
    if separated:
        # Calculate distance between peaks
        distance = np.linalg.norm(np.diff(peaks, axis=0))

        if distance >= original_distance:
            resolution = distance
        

    return resolution


def calculate_resolution(WF, mu_gyro, mu_pitch, inverter, iters, 
                         original_distance, map_type='pitch'):
    '''
    Calculate the resolution of a point from the tomography solution xHat.

    Marina Jimenez Comez: mjimenez37@us.es

    Parameters
    ----------

    WF : xarray.DataArray
        Wave function.
    mu_gyro : list
        Gyro values.
    mu_pitch : list
        Pitch values.
    inverter : str
        Inverter to use: kackzmarz, descent, cimmino.
    iters : int
        Number of iterations.
    original_distance : float
        Original distance between the peaks.
    map_type : str
        Type of resolution map to calculate. Options are 'pitch' and 'gyro'.

    '''

    # Generate synthetic signal
    x, y = synthetic_signal.create_synthetic_delta(WF, mu_gyro, mu_pitch,
                                                    noise_level = 0.1,
                                                    background_level = 0.01,
                                                    seed=0)
    
    if y.max() == 0:
        return 0
    
        # Tomography
    tomo = ss.tomography(WF, y)
    x0 = np.zeros(tomo.s1D.shape)
    if inverter == 'descent':
        tomo.coordinate_descent_solve(x0, iters,  damp = 0.1, 
                                  relaxParam = 1)
    elif inverter == 'kaczmarz':
        tomo.kaczmarz_solve(x0, iters,  damp = 0.1, relaxParam = 1)
    elif inverter == 'cimmino':
        tomo.cimmino_solve(x0, iters,  damp = 0.1, relaxParam = 1)
        
    # Calculate resolution of the point
    xHat = tomo.inversion[inverter].F.isel(alpha = 0).copy()
    resolution = calculate_resolution_point(xHat, original_distance, 
                                                map_type=map_type)
    return resolution




def calculate_resolution_map(WF, inverter, window, maxiter, map_type='pitch'):
    '''
    Calculate the resolution map from the tomography solution xHat.

    Marina Jimenez Comez:

    Parameters
    ----------
     W: xarray.DataArray
        Wave function.
    inverter : str
        Inverter to use: kackzmarz, descent, cimmino.
    window : list
        Window for the coordinate descent algorithm and for resolution_map.
    maxiter : int
        Maximum number of iterations.
    map_type : str
        Type of resolution map to calculate. Options are 'pitch' and 'gyro'.

    '''
    # Generate grid
    # gyroradius for x and pitch for y
    r_values = WF.y.values
    p_values = WF.x.values

    r_liminf = r_values [r_values >= window[2]]
    p_liminf = p_values[p_values >= window[0]]

    r_selected = r_liminf[r_liminf <= window[3]]
    p_selected = p_liminf[p_liminf <= window[1]]

    resolution_mapXR = xr.DataArray(np.nan*np.empty((len(p_selected), len(r_selected))), 
                                coords=[('x', p_selected), ('y', r_selected)])

    if map_type == 'gyro':
        for pitch in p_selected:
            for gyro_i in np.arange(0,len(r_selected)):
                k = 1 # grid separation
                resolution = 0
                while resolution == 0:
                    if gyro_i+k >= len(r_liminf):
                        break
                    mu_gyro = [r_selected[gyro_i], r_liminf[gyro_i+k]]
                    mu_pitch = [pitch, pitch]
                    original_distance = r_liminf[gyro_i+k] - r_selected[gyro_i]
                    resolution = calculate_resolution(WF, mu_gyro, mu_pitch, 
                                                  inverter, maxiter, 
                                                  original_distance, 
                                                  map_type=map_type)
                    k += 1 
                resolution_mapXR.loc[dict(x=mu_pitch[0], y=mu_gyro[0])] = resolution
                

                

    else:
        for gyro in r_selected:
            for pitch_i in np.arange(0,len(p_selected)):
                k = 1
                resolution = 0
                while resolution == 0:
                    if pitch_i+k >= len(p_liminf):
                        break
                    mu_gyro = [gyro, gyro]
                    mu_pitch = [p_selected[pitch_i], p_liminf[pitch_i+k]]
                    original_distance = p_liminf[pitch_i+k] - p_selected[pitch_i]
                    resolution = calculate_resolution(WF, mu_gyro, mu_pitch, 
                                                  inverter, maxiter, 
                                                  original_distance, 
                                                  map_type=map_type)
                    k += 1
                resolution_mapXR.loc[dict(x=mu_pitch[0], y=mu_gyro[0])] = resolution


    
    return resolution_mapXR

                
def get_peaks_sigma(xHat,alpha=20,size=10):
    '''
    Get the peaks of the tomography solution xHat.

    Marina Jimenez Comez:

    Parameters
    ----------
    xHat : xarray.DataArray
        Tomography solution.
    alpha : float
        Alpha value. Hyperparameter to determine the peaks.
    size : int
        Size of the peak.

    Returns
    -------
    pitch_max : np.array
        Pitch of the maximum.
    gyro_max : np.array
        Gyro of the maximum.

    '''

    i_out = []
    j_out = []
    image_temp = copy.deepcopy(xHat.values)
    sigma = np.std(image_temp)
    while True:
        k = np.argmax(image_temp)
        j,i = np.unravel_index(k, image_temp.shape)
        if(image_temp[j,i] >= alpha*sigma):
            i_out.append(i)
            j_out.append(j)
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv,yv = np.meshgrid(x,y)
            image_temp[yv.clip(0,image_temp.shape[0]-1),
                                   xv.clip(0,image_temp.shape[1]-1) ] = 0
        else:
            break

        pitch_max = xHat.x[j_out]
        gyro_max = xHat.y[i_out]
    return pitch_max, gyro_max   



    



    