import logging
import datetime
import numpy as np
import math
import ScintSuite as ss
import ScintSuite.LibData.TCV.Equilibrium as TCV_equilibrium
from ScintSuite.decorators import deprecated
from ScintSuite._Mapping._Calibration import CalParams
from scipy.interpolate import griddata
from lmfit import Parameters
from lmfit import Model
from scipy import special
logger = logging.getLogger('ScintSuite.MappingCommon')
try:
    import lmfit
except (ImportError, ModuleNotFoundError):
    logger.warning('10: You cannot calculate resolutions')


shot = 80617
time_mag = 0.5
tn1 = 0.005
tn2 = 0.020 
t1 = 0.510
t2 = 0.514

# Load the strike map, i.e. the map that relates pixels to E, Lambda 
smap=ss.mapping.StrikeMap('FILD',file='/home/poley/NoTivoli/SINPA/runs/80617@0.450_ul/results/80617@0.450_ul.map')

# Load the video of the camera and read the desired frames 
vid = ss.vid.FILDVideo(file='/videodata/pcfild004/data/fild002/80617.mat')

# - Background substraction: 
substract_noise = True
if substract_noise:
    it1 = np.argmin(np.abs(vid.exp_dat.t.values - tn1))
    it2 = np.argmin(np.abs(vid.exp_dat.t.values - tn2))
    frame = vid.exp_dat['frames'].isel(t=slice(it1, it2)).mean(dim='t')
    
    vid.read_frame(t1=t1, t2=t2)  #first get rid of the frames to be faster
    vid.subtract_noise(frame=frame, flag_copy=False)

# - Filter options:
apply_filter = True  # Flag to apply filter to the frames
# If you want a median one
'''
kind_of_filter = 'median'
options_filter = {
    'size': 3        # Size of the window to apply the filter
}
'''
# If you want a gaussian one
kind_of_filter = 'gaussian'
options_filter = {
    'sigma': 3        # sigma of the gaussian for the convolution (in pixels)
}

if apply_filter:
    vid.filter_frames(kind_of_filter, options_filter)

# Calculate the pixel of the real space,  x-y [mm,mm]
smap.calculate_pixel_coordinates(vid.CameraCalibration)

# Calculate the modulus of  the magnetic field at the detector's pinhole position
xyzPin = np.array([-192.96, 1137.7, 35.4414])*0.001   #get magnetic field for specific slit
Rpin = np.sqrt(xyzPin[0]**2 + xyzPin[1]**2)
zPin = xyzPin[2]
Rinsertion = -17.1
Br, Bz, Bt, bp =  TCV_equilibrium.get_mag_field(shot, Rpin + Rinsertion*0.001, zPin, time_mag)#, use_gdat = True)
modB = np.sqrt(Br**2 + Bz**2 + Bt**2) 

# Convert the variables to Energy - Lambda [keV, -] space (instead of Gyroradius - Pitch angle [cm, Deg.])
smap.calculate_energy(modB)
smap.convert_to_normalized_pitch()
smap.setRemapVariables(('p0', 'e0'), verbose=False)

namex = smap._to_remap[0].name # name of the variable in x position, this is Lambda for us
namey = smap._to_remap[1].name # name of the variable in y position, this is Energy for us

data = vid.exp_dat #counts in the pixel matrix, 864 x 1280 x FramesN
framen = 1 # number or frame selected out of the loaded frames 
frame = data['frames'].values[:, :, framen] #counts in the pixel matrix, 864 x 1280

# Define parameters for the interpolation
par = {
    'ymin': 5.0,      # Minimum gyroradius [in cm]
    'ymax': 60.0,     # Maximum gyroradius [in cm]
    'dy': (60-5)/200.,        # Interval of the gyroradius [in cm]
    'xmin': -1,     # Minimum pitch angle [in degrees]
    'xmax': 1,     # Maximum pitch angle [in degrees]
    'dx': 2/200., # Pitch angle interval
    # method for the interpolation
    'method': 2,  # 2 Spline, 1 Linear
    'decimals': 5,
    'smap_folder': '',
    'map': smap,
    'variables_to_remap':('p0', 'e0'),#'pitch', 'energy'),
    'remap_method':  'griddata',
    }
grid_params = {   # parameters for the montacarlo inversion
    'ymin': par['ymin'],
    'ymax': par['ymax'],
    'dy': par['dy'],
    'xmin': par['xmin'],
    'xmax': par['xmax'],
    'dx': par['dx']
}

#### --- Extra interpolation to get the E-Lambda derivatives of each pixel, thanks Benno and Louis --- ####
from scipy.interpolate import RectBivariateSpline


# Create the grid where we want to interpolate which has the camera frame shape 
grid_x, grid_y = np.mgrid[0:frame.shape[1], 0:frame.shape[0]]


# Prepare the grid for griddata interpolation
dummy = np.column_stack((smap._coord_pix['x'], smap._coord_pix['y']))

variables_to_interpolate = ['p0','e0'] 
# Initialize the structure
if smap._grid_interp is None:
    smap._grid_interp = dict.fromkeys(variables_to_interpolate)
    smap._grid_interp['interpolators'] = dict.fromkeys(variables_to_interpolate)
else:  # Update the dict with new fields
    new_dict = dict.fromkeys(variables_to_interpolate)
    smap._grid_interp.update(new_dict)
    smap._grid_interp['interpolators'].update(new_dict)

derivatives_x = np.zeros((frame.shape[0],frame.shape[1],2))
derivatives_y = np.zeros((frame.shape[0],frame.shape[1],2))
counter=-1
for variable in variables_to_interpolate:
    try:
        # Retrieve data for the current variable
        counter += 1 
        data = smap._data[variable].data

        # Interpolate scattered data onto the grid using griddata
        grid_data = griddata(dummy, data, (grid_x, grid_y), method='cubic', fill_value=0)

        # Now we have a regular grid, we can use RectBivariateSpline if needed
        # Create a 2D mesh grid for RectBivariateSpline
        unique_x = np.unique(grid_x)
        unique_y = np.unique(grid_y)
        
        # Create the interpolator using RectBivariateSpline on the regular grid
        interpolator = RectBivariateSpline(unique_x, unique_y, grid_data)

        # Store the interpolated data
        smap._grid_interp[variable] = grid_data.T

        # Compute derivatives
        # First derivatives
        derivative_x = interpolator.ev(grid_x.T, grid_y.T, dx=1, dy=0)
        derivative_y = interpolator.ev(grid_x.T, grid_y.T, dx=0, dy=1)

        # Store the derivatives
        derivatives_x[:,:,counter]=derivative_x
        derivatives_y[:,:,counter]=derivative_y
    except Exception as e:
        print("Error interpolating")


##################################
### --- JACOBIAN --- ###
from scipy.ndimage import gaussian_filter

jacobian_matrix = np.zeros((frame.shape[0],frame.shape[1]))
for i in range(0,frame.shape[0]):
    for j in range(0,frame.shape[1]):
        dEx = derivatives_x[i,j,1]
        dLx = derivatives_x[i,j,0]
        dEy = derivatives_y[i,j,1]
        dLy = derivatives_y[i,j,0]
        jacobian_matrix[i, j] = abs(dLx*dEy - dLy*dEx)

jacobian_matrix[abs(jacobian_matrix)<0.00001]=np.nan # To avoid divergence, dividing by zero

## Apply filter to smooth the matrix
jacobian_matrix1 = gaussian_filter(jacobian_matrix, sigma=10)   

## Rescale back the amplitude of the matrix     
scaling = np.sum(np.sum(jacobian_matrix[~np.isnan(jacobian_matrix)])) / np.sum(np.sum(jacobian_matrix1[~np.isnan(jacobian_matrix1)]))        
jacobian_matrix1 = jacobian_matrix1*scaling

### --- Make new frame --- ###
frame1=frame/jacobian_matrix1


###################################
smap.interp_grid(frame.shape, method='linear',
                             MC_number=0,
                             grid_params=grid_params,
                             limitation=100)
# --- 1: Information of the calibration
#Get the coordinates of each pixel 
#x = smap._coord_pix['x'] #postion in pixel in the horizontal axis 
#y = smap._coord_pix['y'] #positon in pixel in the vertical axis
# Get the phase variables at each pixel
pix_grid_x, pix_grid_y = np.mgrid[0:frame.shape[1], 0:frame.shape[0]]
xs = np.column_stack((pix_grid_x.flatten(), pix_grid_y.flatten()))
fx = smap._grid_interp['interpolators'][namex]
fy = smap._grid_interp['interpolators'][namey]
ys = np.column_stack((fx(xs[:, 0], xs[:, 1]),fy(xs[:, 0], xs[:, 1])))

idx_isnotnan = ~np.isnan(ys)
xs = xs[idx_isnotnan[:, 0], :]
ys = ys[idx_isnotnan[:, 0], :]
# --- 2: Remap
# Define edges 
x_edges=np.linspace(0,1.0,150)
y_edges=np.linspace(5,60.0,150)
xcenter = 0.5 * (x_edges[1:] + x_edges[:-1])
ycenter = 0.5 * (y_edges[1:] + y_edges[:-1])
XX, YY = np.meshgrid(xcenter, ycenter, indexing='ij')



z = frame.T.flatten().astype(float)


z = z[idx_isnotnan[:, 0]]
H = griddata(ys, z, (XX.flatten(), YY.flatten()),
                method='linear', fill_value=0).reshape(XX.shape)

#I think we do not need this now
# Normalise H to counts per unit of each axis

#delta_x = xcenter[1] - xcenter[0]
#delta_y = ycenter[1] - ycenter[0]
#H /= delta_x * delta_y

