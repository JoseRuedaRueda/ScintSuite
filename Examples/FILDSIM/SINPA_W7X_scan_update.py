# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:03:05 2022

@author: alevines
"""

import numpy as np
import os
import matplotlib.pylab as plt
import Lib as ss
from Lib.LibMachine import machine
from Lib.LibPaths import Path
paths = Path(machine)
import IPython
import pickle
from matplotlib import cm

def get_normal_vector(p1, p2, p3):
    '''
    '''
    #https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    
    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    
    cp = cp/(cp**2).sum()**0.5
    
    return cp
    
def get_x_points_on_circle(xy1, xy2, r, y, right=True):
    '''
    '''
    center_x, center_y = circle_center(xy1, xy2, r, right=right)
    #IPython.embed()
    return np.sqrt(r**2 - (y - center_y)**2 ) + center_x

def circle_center(xy1, xy2, r, right=True):
    #https://stackoverflow.com/questions/4914098/centre-of-a-circle-that-intersects-two-points
    x1 = xy1[0]
    y1 = xy1[1]
    x2 = xy2[0]
    y2 = xy2[1]
    q = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    if r*2 < q:
        
        r = q/2.
    
    y3 = (y1+y2)/2
    x3 = (x1+x2)/2
    # first circle
    #IPython.embed()
    center_x = x3 + np.sqrt(r**2-(q/2)**2)*(y1-y2)/q 
    center_y = y3 + np.sqrt(r**2-(q/2)**2)*(x2-x1)/q 
    if not right:
        # second circle
        center_x = x3 - np.sqrt(r**2-(q/2)**2)*(y1-y2)/q
        center_y = y3 - np.sqrt(r**2-(q/2)**2)*(x2-x1)/q  

    return center_x, center_y

def get_scint_triangles(head_params,
                   slit_number = 1):
    pinhole_length = head_params["pinhole_length"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    
    scint_width = head_params["scint_width"] 
    scint_height = head_params["scint_height"]
    
    scint_theta = head_params["scint_theta"]
    slit2_height_offset = head_params["slit2_height_offset"]
    
    N_triangles = 2
    xyz_scint = np.zeros((N_triangles,3,3))
    
    xyz_scint[0, 0, 0] = - pinhole_scint_dist
    xyz_scint[0, 0, 1] = -0.5 * pinhole_length
    xyz_scint[0, 0, 2] =  0
    
    xyz_scint[0, 1, 0] = - pinhole_scint_dist + np.tan(scint_theta) * scint_height
    xyz_scint[0, 1, 1] = -0.5 * pinhole_length
    xyz_scint[0, 1, 2] =  -scint_height
    
    xyz_scint[0, 2, 0] = - pinhole_scint_dist + np.tan(scint_theta) * scint_height
    xyz_scint[0, 2, 1] = -0.5 * pinhole_length + scint_width
    xyz_scint[0, 2, 2] = -scint_height
    
    xyz_scint[1, 0, :] = xyz_scint[0, 0, :]
    xyz_scint[1, 1, :] = xyz_scint[0, 2, :]
    
    xyz_scint[1, 2, 0] = - pinhole_scint_dist
    xyz_scint[1, 2, 1] = -0.5 * pinhole_length + scint_width
    xyz_scint[1, 2, 2] = 0
    
    scint_normal = get_normal_vector(xyz_scint[0, 0, :],
                                     xyz_scint[0, 1, :], 
                                     xyz_scint[0, 2, :])
    
    if slit_number == 2:
        xyz_scint[:,:, 2] -= slit2_height_offset
    
    return {'N_triangles':N_triangles,
            'triangle_vertices':xyz_scint, 
            'normal':scint_normal }


def get_slit_backplate_triangles(head_params,
                   slit_number = 1):
    # slit backplate
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    scint_to_pfc_dist = head_params["scint_to_pfc_dist"]
    pinhole_dist = pinhole_scint_dist + scint_to_pfc_dist
    scint_height = head_params["scint_height"]
    
    slit2_height_offset = head_params["slit2_height_offset"]
    
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]

        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])
    
    N_triangles = 2
    xyz_slit_back = np.zeros((N_triangles,3,3))
      
    xyz_slit_back[:, :, 1] = np.ones((N_triangles,3)) *(-0.5 * pinhole_length) 
    
    xyz_slit_back[0, 0, 0] = - pinhole_dist
    xyz_slit_back[0, 0, 2] = 0- scint_height
    
    xyz_slit_back[0, 1, 0] =  0.5 * pinhole_width + 2
    xyz_slit_back[0, 1, 2] =  0- scint_height
    
    xyz_slit_back[0, 2, 0] =  0.5 * pinhole_width + 2
    xyz_slit_back[0, 2, 2] =  0
    
    xyz_slit_back[1, 0, :] = xyz_slit_back[0, 0, :]
    xyz_slit_back[1, 1, :] = xyz_slit_back[0, 2, :]
    
    xyz_slit_back[1, 2, 0] = - pinhole_dist
    xyz_slit_back[1, 2, 2] = 0
    
    if slit_number == 2:
        xyz_slit_back[:,:,1] = offset - xyz_slit_back[:,:,1]
        xyz_slit_back[:,:, 2] -= slit2_height_offset
    
    return {'N_triangles':N_triangles,
            'triangle_vertices':xyz_slit_back }

def get_slit_frontplate_triangles(head_params,
                   slit_number = 1):
    #slit frontplate
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        front_slit_theta = head_params["front_slit_theta"]
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
        front_slit_theta = head_params["front_slit_theta_2"]
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])
        
    slit2_height_offset = head_params["slit2_height_offset"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    
    N_triangles = 2
    xyz_slit_front = np.zeros((N_triangles,3,3))
    
    xyz_slit_front[0, 0, 0] = - pinhole_scint_dist
    xyz_slit_front[0, 0, 1] = slit_length
    xyz_slit_front[0, 0, 2] = - slit_length * np.tan(front_slit_theta)
    
    xyz_slit_front[0, 1, 0] = 0.5 * pinhole_width
    xyz_slit_front[0, 1, 1] = slit_length
    xyz_slit_front[0, 1, 2] = - slit_length * np.tan(front_slit_theta)
    
    xyz_slit_front[0, 2, 0] = 0.5 * pinhole_width
    xyz_slit_front[0, 2, 1] = 0.5 * pinhole_length
    xyz_slit_front[0, 2, 2] = 0
    
    xyz_slit_front[1, 0, 0] = - pinhole_scint_dist
    xyz_slit_front[1, 0, 1] = slit_length
    xyz_slit_front[1, 0, 2] = - slit_length * np.tan(front_slit_theta)
    
    xyz_slit_front[1, 1, 0] = - pinhole_scint_dist
    xyz_slit_front[1, 1, 1] = 0.5 * pinhole_length
    xyz_slit_front[1, 1, 2] = 0
    
    xyz_slit_front[1, 2, 0] = 0.5 * pinhole_width
    xyz_slit_front[1, 2, 1] = 0.5 * pinhole_length
    xyz_slit_front[1, 2, 2] = 0
    
    if slit_number == 2:
        xyz_slit_front[:,:,1] = offset - xyz_slit_front[:,:,1]
        
        xyz_slit_front[:,:, 2] -= slit2_height_offset

        
    
    return {'N_triangles':N_triangles,
            'triangle_vertices':xyz_slit_front }

def get_straight_lateral_triangles(head_params,
                   slit_number = 1):
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        slit_length = head_params["slit_length"]
        slit_height = head_params["slit_height"]
    else:
        slit2_height_offset = head_params["slit2_height_offset"]
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
        slit_height = head_params["slit_height_2"]
        slit_length = head_params["slit_length_2"]
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])

    #slit lateral 1
    N_triangles = 2
    xyz_slit_lateral = np.zeros((N_triangles,3,3))
    
    xyz_slit_lateral[0, :, 0] = np.ones(3) *(-0.5 * pinhole_width)
    xyz_slit_lateral[1, :, 0] = np.ones(3) *( 0.5 * pinhole_width)
    xyz_slit_lateral[:, 0, 1] = -0.5 * pinhole_length
    xyz_slit_lateral[:, 0, 2] = - slit_height
    xyz_slit_lateral[:, 1, 1] = slit_length
    xyz_slit_lateral[:, 1, 2] = 0
    xyz_slit_lateral[:, 2, 1] = - 0.5 * pinhole_length
    xyz_slit_lateral[:, 2, 2] = 0
    
    if slit_number == 2:
        xyz_slit_lateral[:,:,1] = offset - xyz_slit_lateral[:,:,1]
        xyz_slit_lateral[:,:, 2] -= slit2_height_offset
    
    return {'N_triangles': N_triangles,
            'triangle_vertices': xyz_slit_lateral }

def get_tilted_lateral_triangles(head_params,
                   slit_number = 1):
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        slit_length = head_params["slit_length"]
        slit_height = head_params["slit_height"]
        slit_theta = head_params["slit_theta"]
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
        slit_height = head_params["slit_height_2"]
        slit_length = head_params["slit_length_2"]
        slit_theta = head_params["slit_theta_2"]
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])

    #slit lateral 1
    N_triangles = 2
    xyz_slit_lateral = np.zeros((N_triangles,3,3))
    
    xyz_slit_lateral[0, 0, 0] = -0.5 * pinhole_width - np.tan(slit_theta) * slit_height
    xyz_slit_lateral[1, 0, 0] =  0.5 * pinhole_width - np.tan(slit_theta) * slit_height
    xyz_slit_lateral[0, 1:, 0] = np.ones(2) *(-0.5 * pinhole_width)
    xyz_slit_lateral[1, 1:, 0] = np.ones(2) *( 0.5 * pinhole_width)
    xyz_slit_lateral[:, 0, 1] = -0.5 * pinhole_length
    xyz_slit_lateral[:, 0, 2] = - slit_height
    xyz_slit_lateral[:, 1, 1] = slit_length
    xyz_slit_lateral[:, 1, 2] = 0
    xyz_slit_lateral[:, 2, 1] = - 0.5 * pinhole_length
    xyz_slit_lateral[:, 2, 2] = 0
    
    if slit_number == 2:
        xyz_slit_lateral[:,:,1] = offset - xyz_slit_lateral[:,:,1]
    
    return {'N_triangles': N_triangles,
            'triangle_vertices': xyz_slit_lateral }

def get_curved_lateral_triangles(head_params,
                   slit_number = 1):
    '''
    Parameters
    ----------
    Returns
    -------
    None.
    '''
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        slit_length = head_params["slit_length"]
        slit_height = head_params["slit_height"]
        opening_curve_radius = head_params["opening_curve_radius"]
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
        slit_height = head_params["slit_height_2"]
        slit_length = head_params["slit_length_2"]
        opening_curve_radius = head_params["opening_curve_radius_2"]
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])

    n_curve_points = head_params["n_curve_points"]
    
    N_triangles = int(2 * (n_curve_points - 1))
    half = int(N_triangles/2)
    
    yz1 = np.array([-0.5 * pinhole_length, -slit_height])
    yz2 = np.array([slit_length,  0.]) #Directly below slit opening
    
    xyz_slit_lateral = np.zeros((N_triangles, 3, 3))
    
    xyz_slit_lateral[0:half, :, 0] = np.ones(3) *(-0.5 * pinhole_width)
    xyz_slit_lateral[half:N_triangles, :, 0] = np.ones(3) *( 0.5 * pinhole_width)
    
    z_points = np.linspace(-slit_height, 0, n_curve_points)
    y_points = get_x_points_on_circle(yz1, yz2, opening_curve_radius, z_points)
    
    xyz_slit_lateral[:, 0, 1] = - 0.5 * pinhole_length
    xyz_slit_lateral[:, 0, 2] = 0
    
    xyz_slit_lateral[0:half, 1, 1] = y_points[0:n_curve_points-1]
    xyz_slit_lateral[0:half, 1, 2] = z_points[0:n_curve_points-1]
    
    xyz_slit_lateral[0:half, 2, 1] = y_points[1:n_curve_points]
    xyz_slit_lateral[0:half, 2, 2] = z_points[1:n_curve_points]
    
    xyz_slit_lateral[half:,:,1:] = xyz_slit_lateral[0:half,:,1:]
    
    if slit_number == 2:
        xyz_slit_lateral[:,:,1] = offset - xyz_slit_lateral[:,:,1]
    
    return {'N_triangles': N_triangles,
            'triangle_vertices': xyz_slit_lateral }

def get_slit_to_scint_triangles(head_params,
                   slit_number = 1):
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
        slit_length = head_params["slit_length"]
        slit_height = head_params["slit_height"]
        pinhole_scint_dist = head_params["pinhole_scint_dist"]
    else:
        slit2_height_offset = head_params["slit2_height_offset"]
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
        slit_height = head_params["slit_height_2"]
        slit_length = head_params["slit_length_2"]
        pinhole_scint_dist = head_params["pinhole_scint_dist"]
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])

    #slit lateral 1
    N_triangles = 2
    xyz_slit_to_scint = np.zeros((N_triangles,3,3))

    xyz_slit_to_scint[:, 0, 0] = - pinhole_scint_dist
    xyz_slit_to_scint[:, 0, 1] = -0.5 * pinhole_length
    xyz_slit_to_scint[:, 0, 2] = - slit_height
    
    xyz_slit_to_scint[0, 1, 0] = -0.5 * pinhole_width
    xyz_slit_to_scint[0, 1, 1] = -0.5 * pinhole_length
    xyz_slit_to_scint[0, 1, 2] = - slit_height
    
    xyz_slit_to_scint[:, 2, 0] = -0.5 * pinhole_width
    xyz_slit_to_scint[:, 2, 1] = slit_length
    xyz_slit_to_scint[:, 2, 2] = 0
    

    xyz_slit_to_scint[1, 1, 0] = - pinhole_scint_dist
    xyz_slit_to_scint[1, 1, 1] = slit_length
    xyz_slit_to_scint[1, 1, 2] = 0
    
   
    if slit_number == 2:
        xyz_slit_to_scint[:,:,1] = offset - xyz_slit_to_scint[:,:,1]
        xyz_slit_to_scint[:,:, 2] -= slit2_height_offset
        
    return {'N_triangles': N_triangles,
            'triangle_vertices': xyz_slit_to_scint }

def get_pinhole_triangles(head_params,
                   slit_number = 1):
    pinhole_padding = 0.
    pinhole_depth = 2.5
    
    double = head_params["double"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    scint_width = head_params["scint_width"]
    
    scint_to_pfc_dist = head_params["scint_to_pfc_dist"]
    pinhole_dist = pinhole_scint_dist + scint_to_pfc_dist    
    
    # Only used if double is on or slit number is 2
    offset = scint_width - 0.5 * (head_params["pinhole_length"] 
                                + head_params["pinhole_length_2"])
    
    slit2_height_offset = head_params["slit2_height_offset"]
    if double and slit_number != 1:
        # Complain, because if double = True, this only needs to be run for the first slit
        print("The code wasn't designed for this! Turn off double or set slit_number to 1")
        print("Turning slit_number back to 1")
        slit_number = 1
    
    if slit_number == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
    
    if double:
        pinhole_width_2 = head_params["pinhole_width_2"]
        width = max(pinhole_width, pinhole_width_2)
        if pinhole_width == pinhole_width_2:
            N_triangles = 10
        else:
            N_triangles = 14
    else:
        N_triangles = 8
        width = pinhole_width
    
    xyz_pinhole = np.zeros((N_triangles, 3, 3))
    
    xyz_pinhole[0, 0, 0] = - pinhole_dist - pinhole_padding
    xyz_pinhole[0, 0, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[0, 1, 0] = -0.5 * width
    xyz_pinhole[0, 1, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    xyz_pinhole[0, 2, 0] = - pinhole_dist - pinhole_padding
    xyz_pinhole[0, 2, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    
    xyz_pinhole[1, 0, 0] = - pinhole_dist - pinhole_padding
    xyz_pinhole[1, 0, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[1, 1, 0] = -0.5 * width
    xyz_pinhole[1, 1, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[1, 2, 0] = -0.5 * width
    xyz_pinhole[1, 2, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    
    xyz_pinhole[2, 0, 0] = 0.5 * width
    xyz_pinhole[2, 0, 1] = -0.5 * pinhole_length
    xyz_pinhole[2, 1, 0] = 0.5 * width
    xyz_pinhole[2, 1, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[2, 2, 0] = -0.5 * width
    xyz_pinhole[2, 2, 1] = -0.5 * pinhole_length - pinhole_padding
    
    xyz_pinhole[3, 0, 0] = 0.5 * width
    xyz_pinhole[3, 0, 1] = -0.5 * pinhole_length
    xyz_pinhole[3, 1, 0] = -0.5 * width
    xyz_pinhole[3, 1, 1] = -0.5 * pinhole_length
    xyz_pinhole[3, 2, 0] = -0.5 * width
    xyz_pinhole[3, 2, 1] = -0.5 * pinhole_length - pinhole_padding

    xyz_pinhole[4, 0, 0] = 0.5 * width
    xyz_pinhole[4, 0, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[4, 1, 0] = - pinhole_dist + pinhole_padding + pinhole_depth
    xyz_pinhole[4, 1, 1] = -0.5 * pinhole_length - pinhole_padding
    xyz_pinhole[4, 2, 0] = - pinhole_dist + pinhole_padding + pinhole_depth
    xyz_pinhole[4, 2, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    
    xyz_pinhole[5, 0, 0] = - pinhole_dist + pinhole_padding + pinhole_depth
    xyz_pinhole[5, 0, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    xyz_pinhole[5, 1, 0] = 0.5 * width
    xyz_pinhole[5, 1, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    xyz_pinhole[5, 2, 0] = 0.5 * width
    xyz_pinhole[5, 2, 1] = -0.5 * pinhole_length - pinhole_padding

    if double:
        pinhole_length_2 = head_params["pinhole_length_2"]
        
        xyz_pinhole[6, 0, 0] = 0.5 * width
        xyz_pinhole[6, 0, 1] = 0.5 * pinhole_length
        xyz_pinhole[6, 1, 0] = 0.5 * width
        xyz_pinhole[6, 1, 1] = offset - 0.5 * pinhole_length_2
        xyz_pinhole[6, 2, 0] = -0.5 * width
        xyz_pinhole[6, 2, 1] = offset - 0.5 * pinhole_length_2
    
        xyz_pinhole[7, 0, 0] = -0.5 * width
        xyz_pinhole[7, 0, 1] = offset - 0.5 * pinhole_length_2
        xyz_pinhole[7, 1, 0] = -0.5 * width
        xyz_pinhole[7, 1, 1] = 0.5 * pinhole_length
        xyz_pinhole[7, 2, 0] = 0.5 * width
        xyz_pinhole[7, 2, 1] = 0.5 * pinhole_length
        
        xyz_pinhole[8, 0, 0] = 0.5 * width
        xyz_pinhole[8, 0, 1] = offset + 0.5 * pinhole_length_2
        xyz_pinhole[8, 1, 0] = 0.5 * width
        xyz_pinhole[8, 1, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
        xyz_pinhole[8, 2, 0] = -0.5 * width
        xyz_pinhole[8, 2, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    
        xyz_pinhole[9, 0, 0] = -0.5 * width
        xyz_pinhole[9, 0, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
        xyz_pinhole[9, 1, 0] = -0.5 * width
        xyz_pinhole[9, 1, 1] = offset + 0.5 * pinhole_length_2
        xyz_pinhole[9, 2, 0] = 0.5 * width
        xyz_pinhole[9, 2, 1] = offset + 0.5 * pinhole_length_2
        
        if pinhole_width > pinhole_width_2:
            xyz_pinhole[10, 0, 0] = -0.5 * width
            xyz_pinhole[10, 0, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[10, 1, 0] = -0.5 * width
            xyz_pinhole[10, 1, 1] = offset + 0.5 * pinhole_length_2
            xyz_pinhole[10, 2, 0] = -0.5 * pinhole_width_2
            xyz_pinhole[10, 2, 1] = offset + 0.5 * pinhole_length_2
            
            xyz_pinhole[11, 0, 0] = -0.5 * width
            xyz_pinhole[11, 0, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[11, 1, 0] = -0.5 * pinhole_width_2
            xyz_pinhole[11, 1, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[11, 2, 0] = -0.5 * pinhole_width_2
            xyz_pinhole[11, 2, 1] = offset + 0.5 * pinhole_length_2
            
            xyz_pinhole[12, 0, 0] = 0.5 * width
            xyz_pinhole[12, 0, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[12, 1, 0] = 0.5 * width
            xyz_pinhole[12, 1, 1] = offset + 0.5 * pinhole_length_2
            xyz_pinhole[12, 2, 0] = 0.5 * pinhole_width_2
            xyz_pinhole[12, 2, 1] = offset + 0.5 * pinhole_length_2
            
            xyz_pinhole[13, 0, 0] = 0.5 * width
            xyz_pinhole[13, 0, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[13, 1, 0] = 0.5 * pinhole_width_2
            xyz_pinhole[13, 1, 1] = offset - 0.5 * pinhole_length_2
            xyz_pinhole[13, 2, 0] = 0.5 * pinhole_width_2
            xyz_pinhole[13, 2, 1] = offset + 0.5 * pinhole_length_2
            
        elif pinhole_width < pinhole_width_2:
            xyz_pinhole[10, 0, 0] = -0.5 * width
            xyz_pinhole[10, 0, 1] = -0.5 * pinhole_length
            xyz_pinhole[10, 1, 0] = -0.5 * width
            xyz_pinhole[10, 1, 1] = 0.5 * pinhole_length
            xyz_pinhole[10, 2, 0] = -0.5 * pinhole_width
            xyz_pinhole[10, 2, 1] = 0.5 * pinhole_length
            
            xyz_pinhole[11, 0, 0] = -0.5 * width
            xyz_pinhole[11, 0, 1] = -0.5 * pinhole_length
            xyz_pinhole[11, 1, 0] = -0.5 * pinhole_width
            xyz_pinhole[11, 1, 1] = -0.5 * pinhole_length
            xyz_pinhole[11, 2, 0] = -0.5 * pinhole_width
            xyz_pinhole[11, 2, 1] = 0.5 * pinhole_length
            
            xyz_pinhole[12, 0, 0] = 0.5 * width
            xyz_pinhole[12, 0, 1] = -0.5 * pinhole_length
            xyz_pinhole[12, 1, 0] = 0.5 * width
            xyz_pinhole[12, 1, 1] = 0.5 * pinhole_length
            xyz_pinhole[12, 2, 0] = 0.5 * pinhole_width
            xyz_pinhole[12, 2, 1] = 0.5 * pinhole_length
            
            xyz_pinhole[13, 0, 0] = 0.5 * width
            xyz_pinhole[13, 0, 1] = -0.5 * pinhole_length
            xyz_pinhole[13, 1, 0] = 0.5 * pinhole_width
            xyz_pinhole[13, 1, 1] = -0.5 * pinhole_length
            xyz_pinhole[13, 2, 0] = 0.5 * pinhole_width
            xyz_pinhole[13, 2, 1] = 0.5 * pinhole_length
        
        
    else:
        xyz_pinhole[6, 0, 0] = 0.5 * width
        xyz_pinhole[6, 0, 1] = 0.5 * pinhole_length
        xyz_pinhole[6, 1, 0] = 0.5 * width
        xyz_pinhole[6, 1, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
        xyz_pinhole[6, 2, 0] = -0.5 * width
        xyz_pinhole[6, 2, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
    
        xyz_pinhole[7, 0, 0] = -0.5 * width
        xyz_pinhole[7, 0, 1] = -0.5 * pinhole_length  + pinhole_padding +  scint_width
        xyz_pinhole[7, 1, 0] = -0.5 * width
        xyz_pinhole[7, 1, 1] = 0.5 * pinhole_length
        xyz_pinhole[7, 2, 0] = 0.5 * width
        xyz_pinhole[7, 2, 1] = 0.5 * pinhole_length
    
    if slit_number == 2:
        xyz_pinhole[:,:,1] = offset - xyz_pinhole[:,:,1]
        xyz_pinhole[:,:, 2] -= slit2_height_offset
    
    return {'N_triangles':N_triangles,
            'triangle_vertices':xyz_pinhole }

def get_pfc_triangles(head_params,
                   slit_number = 1):
    pinhole_length = head_params["pinhole_length"]
    pinhole_scint_dist = head_params["pinhole_scint_dist"]
    
    scint_width = head_params["scint_width"] 
    scint_height = head_params["scint_height"]
    
    scint_theta = head_params["scint_theta"]
    
    scint_to_pfc_dist = head_params["scint_to_pfc_dist"]
    pinhole_dist = pinhole_scint_dist + scint_to_pfc_dist
    
    N_triangles = 2
    xyz_scint = np.zeros((N_triangles,3,3))
    
    xyz_scint[0, 0, 0] = - pinhole_dist
    xyz_scint[0, 0, 1] = -0.5 * pinhole_length
    xyz_scint[0, 0, 2] =  0
    
    xyz_scint[0, 1, 0] = - pinhole_dist + np.tan(scint_theta) * scint_height
    xyz_scint[0, 1, 1] = -0.5 * pinhole_length
    xyz_scint[0, 1, 2] =  -scint_height
    
    xyz_scint[0, 2, 0] = - pinhole_dist + np.tan(scint_theta) * scint_height
    xyz_scint[0, 2, 1] = -0.5 * pinhole_length + scint_width
    xyz_scint[0, 2, 2] = -scint_height
    
    xyz_scint[1, 0, :] = xyz_scint[0, 0, :]
    xyz_scint[1, 1, :] = xyz_scint[0, 2, :]
    
    xyz_scint[1, 2, 0] = - pinhole_dist
    xyz_scint[1, 2, 1] = -0.5 * pinhole_length + scint_width
    xyz_scint[1, 2, 2] = 0
    
    scint_normal = get_normal_vector(xyz_scint[0, 0, :],
                                     xyz_scint[0, 1, :], 
                                     xyz_scint[0, 2, :])
    
    return {'N_triangles':N_triangles,
            'triangle_vertices':xyz_scint, 
            'normal':scint_normal }

def write_geometry_files(root_dir = '/cobra/u/elev/SINPA/Geometry/'
                  , scan_name = ''
                  , head_params = {}
                  , slit = 1):
    '''
    Parameters
    ----------
    pinhole_length : TYPE, optional
        DESCRIPTION. The default is 0.2.
    pinhole_width : TYPE, optional
        DESCRIPTION. The default is 0.1.
    pinhole_scint_dist : TYPE, optional
        DESCRIPTION. The default is 0.5.
    slit_height : TYPE, optional
        DESCRIPTION. The default is 1.0.
    slit_length : TYPE, optional
        DESCRIPTION. The default is 5..
    scint_width : TYPE, optional
        DESCRIPTION. The default is 10..
    scint_height : TYPE, optional
        DESCRIPTION. The default is 10..
    
    Returns
    -------
    None.
    '''
    
    scan_folder = scan_name + '/'
    
    directory = os.path.join(root_dir, scan_folder)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Making directory')
    
    double = head_params["double"]
    opening_curve_radius = head_params["opening_curve_radius"]
    opening_curve_radius_2 = head_params["opening_curve_radius_2"]
    front_slit_theta = head_params["front_slit_theta"]
    
    front_slit_theta_2 = head_params["front_slit_theta_2"]
    slit2_height_offset = head_params["slit2_height_offset"]
    
    dx = 10
    
    #scintillator_plate
    
    scint_plate = get_scint_triangles(head_params, slit_number = slit)
    
    # Write scintillator to file
    scint_filename = directory + 'Element2.txt'
    
    f = open(scint_filename, 'w')
    f.write('Scintillator file for SINPA FILDSIM\n')
    f.write('Scan name is ' + scan_name + '\n')
    f.write('File by A. LeViness' + '\n')
    
    f.write('2  ! Kind of plate\n')
    f.write(str(scint_plate['N_triangles']) + '  ! Number of triangles\n')
    
    # convert all coordinates to m  
    for i in range(scint_plate['N_triangles']):
        for j in range(3):
            f.write(str(np.round((scint_plate['triangle_vertices'][i, j, 0] + dx) * 0.01,6)) + ' '
                    + str(np.round(scint_plate['triangle_vertices'][i, j, 1] * 0.01,6)) + ' '
                    + str(np.round(scint_plate['triangle_vertices'][i, j, 2] * 0.01,6)) + '\n')
    f.close()
    
    # Gather collimator triangles
    triangles = []
    N_triangles = 0
    
    if double or slit == 1:
        pinhole = get_pinhole_triangles(head_params)
        triangles.append(pinhole)
        
        #slit_1 = get_slit_triangles(head_params)
        #triangles.append(slit_1)
        
        pfc_front_plate = get_pfc_triangles(head_params)
        #triangles.append(pfc_front_plate)

        xyz_slit_to_scint = get_slit_to_scint_triangles(head_params)
        triangles.append(xyz_slit_to_scint)
    
        back_plate = get_slit_backplate_triangles(head_params)
        triangles.append(back_plate)
    
        if slit_theta:
            laterals = get_tilted_lateral_triangles(head_params)
        elif opening_curve_radius:
            laterals = get_curved_lateral_triangles(head_params)
        else:
            laterals = get_straight_lateral_triangles(head_params)
    
        triangles.append(laterals)
    
        N_triangles += (pinhole['N_triangles'] #+ slit_1['N_triangles'] 
                      + back_plate['N_triangles'] + laterals['N_triangles']
                      + xyz_slit_to_scint['N_triangles']
                      #+ pfc_front_plate['N_triangles']
                      )
    
        if front_slit_theta:
            front_plate = get_slit_frontplate_triangles(head_params)
            triangles.append(front_plate)
            N_triangles += front_plate['N_triangles']
    
    if double or slit == 2:
        
        if not double:
            pinhole = get_pinhole_triangles(head_params,slit_number=2)
            triangles.append(pinhole)
            N_triangles += pinhole['N_triangles']
        
        # slit_2 = get_slit_triangles(head_params, slit_number=2)
        # triangles.append(slit_2)

        pfc_front_plate = get_pfc_triangles(head_params,slit_number=2)
        #triangles.append(pfc_front_plate)

        xyz_slit_to_scint = get_slit_to_scint_triangles(head_params,slit_number=2)
        triangles.append(xyz_slit_to_scint)
    
        back_plate_2 = get_slit_backplate_triangles(head_params, slit_number=2)
        triangles.append(back_plate_2)
        
        if slit_theta_2:
            laterals_2 = get_tilted_lateral_triangles(head_params, slit_number=2)
        elif opening_curve_radius_2:
            laterals_2 = get_curved_lateral_triangles(head_params, slit_number=2)
        else:
            laterals_2 = get_straight_lateral_triangles(head_params, slit_number=2)
            
        triangles.append(laterals_2)
    
        N_triangles += (back_plate_2['N_triangles'] 
                      + laterals_2['N_triangles']
                      #+ pfc_front_plate['N_triangles']
                      + xyz_slit_to_scint['N_triangles'] )
    
        if front_slit_theta_2:
            front_plate_2 = get_slit_frontplate_triangles(head_params, slit_number = 2)
            triangles.append(front_plate_2)
            N_triangles += front_plate_2['N_triangles']
    
    collimator_filename = directory + 'Element1.txt'
    
    if slit == 1:
        loc = 'left'
    elif slit == 2:
        loc = 'right'
    
    f = open(collimator_filename, 'w')
    f.write('Collimator file for SINPA FILDSIM\n')
    f.write('Scan name is ' + scan_name + '\n')
    f.write('Double slit is ' + str(double) + ', slit location is on the ' + loc +'\n')
    
    f.write('0  ! Kind of plate\n')
    f.write(str(N_triangles) + '  ! Number of triangles\n')
    
    for plate in triangles:
        for i in range(plate['N_triangles']):
            for j in range(3):
                #convert all to m
                f.write(str(np.round((plate['triangle_vertices'][i, j, 0] + dx) * 0.01,6)) + ' '
                        + str(np.round(plate['triangle_vertices'][i, j, 1] * 0.01,6)) + ' '
                        + str(np.round(plate['triangle_vertices'][i, j, 2] * 0.01,6)) + '\n')
    f.close()
    rPinZ = 0
    if slit == 1:
        rPinY = 0
    else:
        offset = head_params["scint_width"] - 0.5 * (head_params["pinhole_length"]
                                                   + head_params["pinhole_length_2"])
        rPinY = offset
        rPinZ = -slit2_height_offset
    
    # Write the file "ExtraGeometryParams"
    if slit == 1:
        pinhole_length = head_params["pinhole_length"]
        pinhole_width = head_params["pinhole_width"]
    else:
        pinhole_length = head_params["pinhole_length_2"]
        pinhole_width = head_params["pinhole_width_2"]
    
    scint_norm = scint_plate['normal']
    scint_norm[0] = -1 * scint_norm[0]
    
    ps = np.zeros(3)
    
    ps[0] = - head_params["pinhole_scint_dist"] + 10
    ps[1] = - head_params["pinhole_length"]/2
    ps[2] = 0
    
    rot = ss.sinpa.geometry.calculate_rotation_matrix(scint_norm,verbose=False)
    
    extra_filename = directory + 'ExtraGeometryParams.txt'
    nGeomElements = 2
    # make sure to convert all to m
    f = open(extra_filename,'w')
    f.write('&ExtraGeometryParams   ! Namelist with the extra geometric parameters\n')
    f.write('  nGeomElements = ' + str(nGeomElements) + '\n')
    f.write('  ! Pinhole\n')
    f.write('  rPin(1) = ' + (str(np.round(dx * 0.01,6))) 
            + ',        ! Position of the pinhole XYZ\n')
    f.write('  rPin(2) = ' + (str(np.round(rPinY * 0.01,6))) + ',\n')
    f.write('  rPin(3) = ' + (str(np.round(rPinZ * 0.01,6))) + ',\n')
    f.write('  pinholeKind = 1     ! 0 = Circular, 1 = rectangle\n')
    f.write('  d1 = ' + (str(np.round(pinhole_width * 0.01,6))) + '  ! Pinhole radius, or size along u1 (in m)\n')
    f.write('  d2 = ' + (str(np.round(pinhole_length * 0.01,6))) + '   ! Size along u2, not used if we have a circular pinhole\n\n')
    f.write('  ! Unitary vectors:\n')
    f.write('  u1(1) =  1.0\n')
    f.write('  u1(2) =  0.0\n')
    f.write('  u1(3) =  0.0\n\n')
    f.write('  u2(1) =  0.0\n')
    f.write('  u2(2) =  1.0\n')
    f.write('  u2(3) =  0.0\n\n')
    f.write('  u3(1) =  0.0   ! Normal to the pinhole plane\n')
    f.write('  u3(2) =  0.0\n')
    f.write('  u3(3) =  1.0\n\n')
    f.write('  ! Reference system of the Scintillator:\n')
    f.write('  ps(1) =  ' + (str(np.round(ps[0] * 0.01,6))) + '\n')
    f.write('  ps(2) =  ' + (str(np.round(ps[1] * 0.01,6))) + '\n')
    f.write('  ps(3) =  ' + (str(np.round(ps[2] * 0.01,6))) + '\n\n')
    # f.write('  ScintNormal(1) =  ' + (str(np.round(scint_norm[0],4))) + '   ! Normal to the scintillator\n')
    # f.write('  ScintNormal(2) =  ' + (str(np.round(scint_norm[1],4))) + '\n')
    # f.write('  ScintNormal(3) =  ' + (str(np.round(scint_norm[2],4))) + '\n\n')
    f.write('  rotation(1,1) = ' + (str(np.round(rot[0,0],4))) + '\n')
    f.write('  rotation(1,2) = ' + (str(np.round(rot[0,1],4))) + '\n')
    f.write('  rotation(1,3) = ' + (str(np.round(rot[0,2],4))) + '\n')
    f.write('  rotation(2,1) = ' + (str(np.round(rot[1,0],4))) + '\n')
    f.write('  rotation(2,2) = ' + (str(np.round(rot[1,1],4))) + '\n')
    f.write('  rotation(2,3) = ' + (str(np.round(rot[1,2],4))) + '\n')
    f.write('  rotation(3,1) = ' + (str(np.round(rot[2,0],4))) + '\n')
    f.write('  rotation(3,2) = ' + (str(np.round(rot[2,1],4))) + '\n')
    f.write('  rotation(3,3) = ' + (str(np.round(rot[2,2],4))) + '\n\n')
    f.write('/')
    f.close()
    
    
    return

if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # --- Options for running a scan
    # -----------------------------------------------------------------------------
    
    Test = False  #if true don't submit run
    
    run_code = True
    scan = False # Set to "False" to run a single iteration
    scan_param = ['pw'] # , 'pw2', 'pl', 'pl2', 'psd', 'sh', 'sh2', 'sl', 'sl2', 'oc', 'oc2', 'st', 'ct', 'ct2']
    geom_name = 'W7X_test_back'#'W7X_scan_pw_0.08' # Only used if running a single iteration
    
    save_orbits = False
    plot_plate_geometry = True
    plot_3D = True
    
    read_results = not run_code #True
    plot_strike_points = True
    plot_strikemap = True
    plot_orbits = False
    plot_metrics = True
    
    #for best case: choose the scan parameter and value you want to look at!
    best_param = 'pl'
    best_param_value = 3.0
    plot_resolutions = True
    plot_gyro_res= True
    plot_pitch_res= True
    plot_collimating_factor = True
    plot_synthetic_signal= False
    
    # DOUBLE SLIT FUNCTIONS
    # For each array, the first value is for the left slit and the second is for
    # the right slit
    # Examples of what you can do:
    # If double = True and run_slit = [True, False], it will plot both the left
    # and the right slit but only run particles starting at the left slit
    # If double = False and run_slit = [True, True], it will run each of the two slits
    # as if they are the only slits in the geometry
    # You probably don't want to do that if you're doing a scan of a property that
    # only impacts one slit, like if you're scanning the width of the left slit only
    # Reasons to have double = True: if you want to check if the second slit geometry
    # interferes with particles launched from the first slit or vice versa
    # Reasons to have double = False: to reduce the number of triangles and
    # save computation time
    # My advice: leave double off for scans, turn it on once you choose values to
    # check if it makes a difference
    
    double = False # Plot both slits in the geometry files
    run_slit = [True, False] # Run particles starting at one slit or the other?
    read_slit = [True, False] # Read results from the left slit, the right slit, or both?
    
    # Or you can mix-and-match: read one file from one scan and another from another,
    # or have one set file for one slit and show it next to a scan done on the other!
    # The read_slit option is to use if the names of files for both slits are from the same scan
    # Note that if you ran with a double-slit, and then try to mix and match,
    # it'll end up plotting the slits on top of each other!
    # Also, this will not work out if your scintillators are not the same size or angle
    # I'll eventually fix this to let you change the scintillator size or something
    mixnmatch = False
    if mixnmatch:
        # Choose the names of each file you want to combine
        base_names = ['Test','Test_right']
        # base_names = ['scan_pw_0.08', 'scan_pw2_0.08']
        if scan:
            # For below: you are only allowed to do one at a time to avoid a huge
            # number of plots, but if scan_slit is "right," it will pull up
            # the left slit designated by base_names[0], then show you a plot
            # of how it looks next to every scan of the right-most slit
            scan_slit = 'left'
            if scan_slit == 'right':
                read_slit = [False, True]
            else:
                read_slit = [True, False]
    
        
    if run_code:
        if not run_slit[0] and not run_slit[1]:
            print('You need to choose at least one slit!')
    
    if read_results:
        if not read_slit[0] and not read_slit[1]:
            print('You need to choose at least one slit!')
                            
    
    marker_params = [{'markersize':6, 'marker':'o','color':'b'},
                     {'markersize':6, 'marker':'o','color':'m'}]
    line_params =  [{'ls':'solid','color':'k'},
                    {'ls':'solid','color':'w'}]
    mar_params = [{'zorder':3,'color':'k'},
                  {'zorder':3,'color':'w'}]
    
    #[cm]
    pinhole_length  = 0.1#25 #0.2 
    pinhole_width   = 0.08 #0.1
    pinhole_scint_dist = 0.8 #0.6 #0.5
    slit_height     = 0.04 #1.0
    slit_length     = 2.0#6. #5.
    front_slit_theta = 0 #degrees
    scint_width     = 6 #large scintilator
    scint_height    = 6
    
    n_curve_points = 20#2
    opening_curve_radius = 0
    scint_theta  = 0
    slit_theta = 0
    
    # Second slit parameters, currently can't have different pinhole_scint_dist's
    pinhole_length_2  = 0.1 #0.2 
    pinhole_width_2   = 0.08 #0.1
    slit_height_2     = 0.5 #1.0
    slit_length_2     = 1.0#6. #5.
    front_slit_theta_2 = 5 #degrees
    
    opening_curve_radius_2 = 0
    slit_theta_2 = 0
    
    slit2_height_offset = 4
    scint_height    += slit2_height_offset
    
    scint_to_pfc_dist = 0.25
    
    
    if double and (slit_length + slit_length_2 >= scint_width):
        print('Your slits are overlapping! Fix your geometry and try again')
    
    n_markers = int(1e4)
    
    gyro_arrays = [[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.],
                   [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2., 3., 4.]]
    #gyro_arrays = [[5.0, 10.0, 20., 30., 40., 50., 75., 100.],
    #               [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20., 30., 40.]]
    pitch_arrays = [[85., 75., 65., 55., 45., 35., 25., 15., 5.],
                    [95., 105., 115., 125., 135., 145., 155., 165., 175.]]
                    #[185., 175., 165., 155., 135., 125., 115., 105., 95.]]
    #pitch_arrays = [[160., 140., 120., 100., -80., -60., -40., -20.],
    #                [160., 140., 120., 100., 80., 60., 40., 20.]]
    gyrophase_ranges = np.array([[3.1415, 6.283],[3.1415, 6.283]])
    
    # Set n1 and r1 for the namelist, 0 and 0 are defaults, setting to 0.02 and 0.4 gives ~300,000 particles for rl = 0.25 
    # and ~400,000 for 0.5
    n1 = 0.0
    r1 = 0.0
    
    Br, Bz, Bt = -0.658720, 0.279007,	2.294939   #STD
    #Br, Bz, Bt = -0.635491, 0.285405, 2.231590    #KJM  
    #Br, Bz, Bt =  -0.681338, 0.245188, 2.279179  #FTM 
    
    modB = np.sqrt(Br**2 + Bz**2 + Bt**2)
    
    alpha = 0.  #
    beta = 0.
    
    use_aligned_B = True    
    use_rotated_FILD_B = False
    use_ascot_B = False
    ascot_bfield_File = 'std_bfield.pickle'
    ascot_boozer_File = 'std_boozer.pickle'
    
    head_params = {
                   "pinhole_length": pinhole_length,
                   "pinhole_width" : pinhole_width ,
                   "pinhole_scint_dist": pinhole_scint_dist,
                   "slit_height"   : slit_height,
                   "slit_length"   : slit_length,
                   "scint_width"   : scint_width, 
                   "scint_height"  : scint_height,
                   "beta"          : beta,
                   "n_curve_points": n_curve_points,
                   "opening_curve_radius": opening_curve_radius,
                   "scint_theta"   : np.deg2rad(scint_theta),
                   "slit_theta"   : np.deg2rad(slit_theta),
                   "front_slit_theta": np.deg2rad(front_slit_theta),
                   "pinhole_length_2": pinhole_length_2,
                   "pinhole_width_2" : pinhole_width_2,
                   "slit_height_2"   : slit_height_2,
                   "slit_length_2"   : slit_length_2,
                   "slit_theta_2"   : np.deg2rad(slit_theta_2),
                   "opening_curve_radius_2": opening_curve_radius_2,
                   "front_slit_theta_2": np.deg2rad(front_slit_theta_2),
                   "double" : double,
                   "Beta": 0.,
                   "scint_to_pfc_dist" : scint_to_pfc_dist,
                   "slit2_height_offset" : slit2_height_offset
                   }
        
    # -----------------------------------------------------------------------------
    # --- Define Scan Parameters
    # -----------------------------------------------------------------------------
    if scan:
        scan_Parameters = {}
        scan_Parameters['Pinhole width'] = {'scan_str': 'pw',
                                            'scan_param': 'pinhole_width',
                                            'scan_values':np.arange(0.03, 0.15, 0.01),
                                            'scan': False,
                                            'value': pinhole_width}
        scan_Parameters['Pinhole length'] = {'scan_str': 'pl',
                                            'scan_param': 'pinhole_length',
                                        'scan_values':np.arange(0.05, 0.45, 0.025), #[0.05, 0.07, 0.13, 0.19],#
                                        'scan': False,
                                        'value': pinhole_length}
        scan_Parameters['Pinhole width_2'] = {'scan_str': 'pw2',
                                        'scan_param': 'pinhole_width_2',
                                        'scan_values':[0.08],
                                        #'scan_values':np.arange(0.03, 0.15, 0.01),
                                        'scan': False,
                                        'value': pinhole_width_2}
        scan_Parameters['Pinhole length_2'] = {'scan_str': 'pl2',
                                        'scan_param': 'pinhole_length_2',
                                        'scan_values':np.arange(0.05, 0.25, 0.02), #[0.05, 0.07, 0.13, 0.19],#
                                        'scan': False,
                                        'value': pinhole_length_2}
        scan_Parameters['Pinhole scint dist'] = {'scan_str': 'psd',
                                        'scan_param': 'pinhole_scint_dist',
                                        'scan_values': np.arange(0.1, 1.5, 0.1),
                                        'scan': False ,
                                        'value': pinhole_scint_dist}
        scan_Parameters['Slit height'] = {'scan_str': 'sh',
                                        'scan_param': 'slit_height',
                                        'scan_values': np.arange(0.1, 2.5, 0.2),
                                        'scan': False,
                                        'value': slit_height}
        scan_Parameters['Slit length'] = {'scan_str': 'sl',
                                        'scan_param': 'slit_length',
                                        'scan_values': np.arange(0.5, 3.0, 0.5),
                                        'scan': False,
                                        'value': slit_length}
        scan_Parameters['Slit height_2'] = {'scan_str': 'sh2',
                                        'scan_param': 'slit_height_2',
                                        'scan_values': np.arange(0.1, 2.5, 0.2),
                                        'scan': False,
                                        'value': slit_height_2}
        scan_Parameters['Slit length_2'] = {'scan_str': 'sl2',
                                        'scan_param': 'slit_length_2',
                                        'scan_values': np.arange(0.5, 3.0, 0.5),
                                        'scan': False,
                                        'value': slit_length_2}
        scan_Parameters['Beta'] = {'scan_str': 'b',
                                   'scan_param':'Beta',
                                        'scan_values': np.arange(-70, 70, 10),#np.arange(0, 215, 214),
                                        'scan': False,
                                        'value': beta}    
        scan_Parameters['opening_curve_radius'] = {'scan_str': 'oc',
                                        'scan_param': 'outer_curve_radius',
                                        'scan_values': [4],#np.arange(5, 17, 2),
                                        'scan': False,
                                        'value': opening_curve_radius} 
        scan_Parameters['opening_curve_radius_2'] = {'scan_str': 'oc2',
                                        'scan_param': 'outer_curve_radius_2',
                                        'scan_values': [4],#np.arange(5, 17, 2),
                                        'scan': False,
                                        'value': opening_curve_radius_2} 
        scan_Parameters['Slit theta'] = {'scan_str': 'ct',
                                        'scan_values': np.arange(-20, 20, 5),
                                        'scan': False,
                                        'value': slit_theta} 
        scan_Parameters['Slit theta 2'] = {'scan_str': 'ct2',
                                        'scan_values': np.arange(-20, 20, 5),
                                        'scan': False,
                                        'value': slit_theta_2} 
        scan_Parameters['Scint theta'] = {'scan_str': 'st',
                                        'scan_param': 'scint_theta',
                                        'scan_values': [0],#np.arange(-20, 20, 5),
                                        'scan': False,
                                        'value': scint_theta}
        for scan_parameter in scan_Parameters.keys():
            if scan_Parameters[scan_parameter]['scan_str'] in scan_param:
                scan_Parameters[scan_parameter]['scan'] = True
    
    # -----------------------------------------------------------------------------
    # --- Run SINPA FILDSIM
    # -----------------------------------------------------------------------------
                
    if run_code:
        geom_dir = os.path.join(paths.SINPA,'Geometry/')
        nml_options = {
            'config':  {  # parameters
                'runid': '',
                'geomfolder': '',
                'FILDSIMmode': True,
                'nxi': 0,
                'nGyroradius': 0,
                'nMap': n_markers,
                'n1': n1,
                'r1': r1,
                'restrict_mode': True,
                'mapping': True,
                'saveOrbits': save_orbits,
                'saveRatio': 0.01,
                'saveOrbitLongMode': False,
                'runfolder': '',
                'verbose': True,
                'IpBt': -1,        # Sign of toroidal current vs field (for pitch), need to check
                'flag_efield_on': False,  # Add or not electric field
                'save_collimator_strike_points': False,  # Save collimator points
                'backtrace': False  # Flag to backtrace the orbits
                },
            'inputParams': {
                'nGyro': 350,
                'minAngle': 0,
                'dAngle': 0,
                'XI': [],
                'rL': [],
                'maxT': 0.00000006
                },
            }
            
        if not scan:
            geom_names = [geom_name,geom_name + '_right']
            
            #Create magnetic field
            field = ss.simcom.Fields()
                    
            if use_aligned_B:
                # This is negative so the particles will gyrate the way we want
                direction = np.array([0, -1, 0])
                Field = modB * direction
                field.createHomogeneousField(Field, field='B')
            elif use_rotated_FILD_B:
                # Haven't tried this out yet, waiting on updates to the angles
                Field = np.array([Br, Bz, Bt])
                phi, theta = ss.fildsim.calculate_fild_orientation(Br, Bz, Bt, alpha, beta, verbose=False)
                u1 = np.array([1., 0., 0.])
                u2 = np.array([0., 1., 0.])
                u3 = np.array([0., 0., 1.])
                field.createHomogeneousFieldThetaPhi(theta, phi, field_mod = modB,
                                                     field='B', u1=u1, u2=u2, u3=u3,
                                                     IpBtsign = 1., verbose = False)
            else:
                f = open(ascot_bfield_File, 'rb')
                ascot_bfield = pickle.load(f)
                f.close()
                #Field geometry saved in "boozer" structure
                f = open(ascot_boozer_File, 'rb')
                ascot_boozer = pickle.load(f)
                f.close()
                
                field.Bfield['R'] = np.linspace(ascot_boozer['rmin'][0], 
                                       ascot_boozer['rmax'][0], 
                                       ascot_boozer['nr'][0])
                field.Bfield['z'] = np.linspace(ascot_boozer['zmin'][0], 
                                       ascot_boozer['zmax'][0], 
                                       ascot_boozer['nz'][0])   
                
                
                field.Bfield['nR'] = len(field.Bfield['R'])
                field.Bfield['nZ'] = len(field.Bfield['z'])
                
                #Ascot stellarator fields only store data for a single period
                #bfield [idx_R, idx_phi, idx_Z], thus rrepeat along axis = 1
                br = np.repeat(ascot_bfield['br'], 
                               ascot_bfield['toroidalPeriods'],
                               axis = 1)
                bphi = np.repeat(ascot_bfield['bphi'], 
                                 ascot_bfield['toroidalPeriods'],
                                 axis = 1)
                bz = np.repeat(ascot_bfield['bz'], 
                               ascot_bfield['toroidalPeriods'],
                               axis = 1)
                
                field.Bfield['fr'] = np.asfortranarray(br)
                field.Bfield['fz'] = np.asfortranarray(bz)
                field.Bfield['ft'] = np.asfortranarray(bphi)
                
                field.Bfield['nPhi'] = np.shape(br)[1] 
    
                field.Bfield['nTime'] = 0
                field.Bfield['Phimin'] = 0.
                field.Bfield['Phimax'] = 2.*np.pi
                
                field.Bfield['Timemin'] = 0.
                field.Bfield['Timemax'] = 0.
                
                field.bdims = 3
            #Write geometry files
            for i in range(2):
                if run_slit[i]:
                    write_geometry_files(root_dir = geom_dir,
                                         scan_name = geom_names[i],
                                         head_params = head_params,
                                         slit = int(i+1))
                        
                        
                    if not Test:
                        # Create directories
                        runDir = os.path.join(paths.SINPA, 'runs', geom_names[i])
                        inputsDir = os.path.join(runDir, 'inputs')
                        resultsDir = os.path.join(runDir, 'results')
                        os.makedirs(runDir, exist_ok=True)
                        os.makedirs(inputsDir, exist_ok=True)
                        os.makedirs(resultsDir, exist_ok=True)
                        
                        # Set namelist parameters
                        nml_options['config']['runid'] = geom_names[i]
                        nml_options['config']['geomfolder'] = geom_dir + '/' + geom_names[i]
                        nml_options['config']['runfolder'] = runDir
                        nml_options['config']['nxi'] = len(pitch_arrays[i])
                        nml_options['config']['nGyroradius'] = len(gyro_arrays[i])
                        nml_options['inputParams']['XI'] = pitch_arrays[i]
                        nml_options['inputParams']['rL'] = gyro_arrays[i]
                        nml_options['inputParams']['minAngle'] = gyrophase_ranges[i,0]
                        nml_options['inputParams']['dAngle'] = (gyrophase_ranges[i,1]
                                                                  - gyrophase_ranges[i,0])
                                                
                        #Make field
                        fieldFileName = os.path.join(inputsDir, 'field.bin')
                        fid = open(fieldFileName, 'wb')
                        field.tofile(fid)
                        fid.close()
    
                        
                        # Create namelist
                        ss.sinpa.execution.write_namelist(nml_options)
                        
                        # Missing a step: create B field!!
                        # Check the files
                        ss.sinpa.execution.check_files(nml_options['config']['runid'])
                        # Launch the simulations
                        ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = True)
                        
            if plot_plate_geometry:
                for i in range(2):
                    if run_slit[i]:
                        geomID = geom_names[i]
                        Geometry = ss.simcom.Geometry(GeomID=geomID)
                        if plot_3D:
                            Geometry.plot3Dfilled(element_to_plot = [0,2])
                        Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], 
                                              plot_pinhole = False, surface_params = {'alpha': 1.0})
                        Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [0,2], plot_pinhole = False)
    
        else:
            for scan_parameter in scan_Parameters.keys():
                if scan_Parameters[scan_parameter]['scan']:
                    for value in scan_Parameters[scan_parameter]['scan_values']:
                        '''
                        Loop over scan paramater
                        '''
                        name = scan_Parameters[scan_parameter]['scan_param']
                        temp = head_params[name]
                        head_params[name] = value
                        if name == 'Beta':
                            beta= value
                        string_mod = '_%s_%s' %(scan_Parameters[scan_parameter]['scan_str'], 
                                                f'{float(value):g}')
                        run_names = ['W7X_scan' + string_mod, 'W7X_scan' + string_mod + '_right']
    
                        #Create magnetic field
                        field = ss.simcom.Fields()
                                            
                        if use_aligned_B:
                            direction = np.array([0, -1, 0])
                            if name == 'Beta':
                                direction = np.array([0, -1*np.cos(np.deg2rad(value)), 1*np.sin(np.deg2rad(value))])
                            Field = modB * direction
                            field.createHomogeneousField(Field, field='B')
                
                        else:
                            Field = np.array([Br, Bz, Bt])
                            phi, theta = ss.fildsim.calculate_fild_orientation(Br, Bz, Bt, 
                                                                               alpha, beta, 
                                                                               verbose=False)
                            u1 = np.array([1., 0., 0.])
                            u2 = np.array([0., 1., 0.])
                            u3 = np.array([0., 0., 1.])
                            field.createHomogeneousFieldThetaPhi(theta, phi, field_mod = modB,
                                                                 field='B', u1=u1, u2=u2, u3=u3,
                                                                  verbose = False)
                
                        for i in range(2):
                            if run_slit[i]:
                                write_geometry_files(root_dir = geom_dir,
                                                     scan_name = run_names[i],
                                                     head_params = head_params,
                                                     slit = int(i+1))
                        
                                if not Test:
                                    # Create directories
                                    runDir = os.path.join(paths.SINPA, 'runs', run_names[i])
                                    inputsDir = os.path.join(runDir, 'inputs')
                                    resultsDir = os.path.join(runDir, 'results')
                                    os.makedirs(runDir, exist_ok=True)
                                    os.makedirs(inputsDir, exist_ok=True)
                                    os.makedirs(resultsDir, exist_ok=True)
                                    
                                    # Set namelist parameters
                                    nml_options['config']['runid'] = run_names[i]
                                    nml_options['config']['geomfolder'] = (geom_dir + '/' + run_names[i])
                                    nml_options['config']['runfolder'] = runDir
                                    nml_options['config']['nxi'] = len(pitch_arrays[i])
                                    nml_options['config']['nGyroradius'] = len(gyro_arrays[i])
                                    nml_options['inputParams']['XI'] = pitch_arrays[i]
                                    nml_options['inputParams']['rL'] = gyro_arrays[i]
                                    nml_options['inputParams']['minAngle'] = gyrophase_ranges[i,0]
                                    nml_options['inputParams']['dAngle'] = (gyrophase_ranges[i,1]
                                                                      - gyrophase_ranges[i,0])
                         
                                    #Make field
                                    fieldFileName = os.path.join(inputsDir, 'field.bin')
                                    fid = open(fieldFileName, 'wb')
                                    field.tofile(fid)
                                    fid.close()
                            
                                    # Create namelist
                                    ss.sinpa.execution.write_namelist(nml_options)
                                
                                    # Missing a step: create B field!!
                                    # Check the files
                                    ss.sinpa.execution.check_files(nml_options['config']['runid'])
                                    # Launch the simulations
                                    ss.sinpa.execution.executeRun(nml_options['config']['runid'], queue = True)
                        
                                head_params[name] = temp
                        
                        if plot_plate_geometry:
                            geomID = run_names[0]
                            Geometry = ss.simcom.Geometry(GeomID=geomID)
                            if plot_3D:
                                Geometry.plot3Dfilled(element_to_plot = [0,2], plot_pinhole = False)
                            Geometry.plot2Dfilled(view = 'XY', element_to_plot = [0,2], plot_pinhole = False)
                            Geometry.plot2Dfilled(view = 'Scint', element_to_plot = [0,2], plot_pinhole = False)
    
    
    # -----------------------------------------------------------------------------
    # --- Section 2: Analyse the results
    # -----------------------------------------------------------------------------
    if read_results:
        if not scan:
            if mixnmatch:
                runid = base_names
            else:
                runid = [geom_name,geom_name + '_right']
            strike_points_file = ['','']
            Smap = [[],[]]
            p0 = [75, 120]
            
            for i in range(2):
                if read_slit[i] or mixnmatch:
                    runDir = os.path.join(paths.SINPA, 'runs', runid[i])
                    inputsDir = os.path.join(runDir, 'inputs/')
                    resultsDir = os.path.join(runDir, 'results/')
                    base_name = resultsDir + runid[i]
                    smap_name = base_name + '.map'
                    
                    # Load the strike map
                    Smap[i] = ss.mapping.StrikeMap('FILD', file=smap_name)
                    try:
                        Smap[i].load_strike_points()
                    except:
                        print('Strike map ' + str(i+1) + ' could not be loaded')
                        continue
    
                    
            if plot_plate_geometry:
                fig, ax = plt.subplots()
                orb = []
                    
                ax.set_xlabel('Y [cm]')
                ax.set_ylabel('Z [cm]')
                ax.set_title('Camera view (YZ plane)')
                
                # If you want to mix and match, it assumes they have the same scintillator
                if mixnmatch:
                    Geometry1 = ss.simcom.Geometry(GeomID=runid[0])
                    Geometry2 = ss.simcom.Geometry(GeomID=runid[1])
                    
                    Geometry1.plot2Dfilled(ax=ax, view = 'YZ', element_to_plot = [0,2],
                                           plot_pinhole = False)
                    Geometry2.plot2Dfilled(ax=ax, view = 'YZ', element_to_plot = [0,2],
                                           plot_pinhole = False)
                else:
                    if read_slit[0]:
                        Geometry = ss.simcom.Geometry(GeomID=runid[0])
                    else:
                        Geometry = ss.simcom.Geometry(GeomID=runid[1])
                        
                    Geometry.plot2Dfilled(ax=ax, view = 'YZ', element_to_plot = [0,2],
                                          plot_pinhole = False)
                    
                for i in range(2):
                    if read_slit[i] or mixnmatch:
                        if plot_strike_points:
                            #IPython.embed()
                            Smap[i].strike_points.scatter(ax=ax, per=0.1, xscale=100.0, yscale=100.0,
                                                          mar_params = mar_params[i])
                                
                        if plot_strikemap:
                            Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                              line_params=line_params[i], factor=100.0, labels=True)
                                
                        if plot_orbits:
                            orb.append(ss.sinpa.orbits(runID=runid[i]))
                            orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                            #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                        
                    else:
                        orb.append([])
                          
                fig.show()
                            
                        
                if plot_3D:
                    if mixnmatch:
                        Geometry1.plot3Dfilled(element_to_plot = [0,2])
                        ax3D = plt.gca()
                        Geometry2.plot3Dfilled(element_to_plot = [0,2], ax=ax3D)
                    else:
                        Geometry.plot3Dfilled(element_to_plot = [0,2])
                        ax3D = plt.gca()
                    
                    if plot_orbits:   
                        for i in range(2):
                            if read_slit[i] or mixnmatch:
                                # plot in red the ones colliding with the scintillator
                                orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, factor=100.0)
                                # plot in blue the ones colliding with the collimator
                                orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, factor=100.0)
                                # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                # the scintillator and nothing is wrong with the code
                                orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, factor=100.0)
    
            if plot_metrics:
                for i in range(2):
                    if read_slit[i] or mixnmatch:
                        Smap[i].calculate_resolutions(min_statistics=100)
              
                        if plot_resolutions:
                            Smap[i].plot_resolutions()
                            
                        if plot_gyro_res:
                            Smap[i].plot_resolution_fits(var='Gyroradius',
                                                         pitch=p0[i],
                                                         kind_of_plot='normal',
                                                         include_legend=True)
                            plt.gcf().show()
                        if plot_pitch_res:
                            Smap[i].plot_resolution_fits(var='Pitch',
                                                         gyroradius=1,
                                                         kind_of_plot='normal',
                                                         include_legend=True)
                            plt.gcf().show()
                        if plot_collimating_factor:
                            Smap[i].plot_collimator_factor()
                            plt.gcf().show()
            #IPython.embed()
            #del Smap
                        
        elif scan:
            orb = [[],[]]
            if mixnmatch:
                # Load the strike map of the slit that isn't being scanned
                if scan_slit == 'right':
                    slit = 0
                else:
                    slit = 1
                constant_slit = base_names[slit]
                runDir = os.path.join(paths.SINPA, 'runs', constant_slit)
                inputsDir = os.path.join(runDir, 'inputs/')
                resultsDir = os.path.join(runDir, 'results/')
                base_name = resultsDir + constant_slit
                smap_name = base_name + '.map'
                Smap[slit] = ss.mapping.StrikeMap('FILD', file=smap_name)
                                
                # Load the strike points used to calculate the map
                try:
                    Smap[slit].load_strike_points()
                except:
                    print('Strike map ' + str(slit) + ' could not be loaded')
                    
                if plot_plate_geometry:
                    Geometry1 = ss.simcom.Geometry(GeomID=constant_slit)
                
                if plot_orbits:
                    orb[slit] = ss.sinpa.orbits(runID=constant_slit)
                
            for scan_parameter in scan_Parameters.keys():
                if scan_Parameters[scan_parameter]['scan']:
                    min_gyro, max_gyro = [[],[]], [[],[]]
                    min_gyro_p0, max_gyro_p0 = [[],[]], [[],[]]
                    gyro_1_res, gyro_2_res, gyro_3_res = [[],[]], [[],[]], [[],[]]
                    min_pitch, max_pitch = [[],[]], [[],[]]
                    min_gyro_res, max_gyro_res = [[],[]], [[],[]]
                    min_pitch_res, max_pitch_res, pitch_p0_res = [[],[]], [[],[]], [[],[]]
                    # p0 is 75 for left slit, 125 for right slit
                    p0 = [75,120]
                
                    avg_collimating_factor, pitch_p0_gyro_1_collimating_factor = [[],[]], [[],[]]
                
                    for value in scan_Parameters[scan_parameter]['scan_values']:
                        ## Loop over scan variables
                        Smap = [[],[]]
                        scan_Parameters[scan_parameter]['value'] = value
                        
                        string_mod = string_mod = '_%s_%s' %(scan_Parameters[scan_parameter]['scan_str'], 
                                                f'{float(value):g}')
                    
                        # Load the result of the simulation
                        runid = ['W7X_scan' + string_mod, 'W7X_scan' + string_mod + '_right']
                    
                        for i in range(2):
                            if read_slit[i]:
                                runDir = os.path.join(paths.SINPA, 'runs', runid[i])
                                inputsDir = os.path.join(runDir, 'inputs/')
                                resultsDir = os.path.join(runDir, 'results/')
                                base_name = resultsDir + runid[i]
                                smap_name = base_name + '.map'
                    
                                # Load the strike map
                                Smap[i] = ss.mapping.StrikeMap('FILD', file=smap_name)
                                
                                # Load the strike points used to calculate the map
                                try:
                                    Smap[i].load_strike_points()
                                except:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                    min_gyro_res[i].append(np.nan)
                                    max_gyro_res[i].append(np.nan)
                                    pitch_p0_res[i].append(np.nan)
                                    gyro_1_res[i].append(np.nan)
                                    gyro_2_res[i].append(np.nan)
                                    gyro_3_res[i].append(np.nan)
                                    min_pitch_res[i].append(np.nan)
                                    max_pitch_res[i].append(np.nan)
                                    avg_collimating_factor[i].append(np.nan)
                                    pitch_p0_gyro_1_collimating_factor[i].append(np.nan)
                                    print('Strike map ' + str(i+1) + ' could not be loaded')
                                    continue
                                
                        if plot_plate_geometry:
                            fig, ax = plt.subplots()
                    
                            ax.set_xlabel('Y [cm]')
                            ax.set_ylabel('Z [cm]')
                            ax.set_title('Camera view (YZ plane)')
                        
                            if read_slit[0]:
                                Geometry = ss.simcom.Geometry(GeomID=runid[0])
                            else:
                                Geometry = ss.simcom.Geometry(GeomID=runid[1])
                        
                            Geometry.plot2Dfilled(ax=ax, view = 'YZ', element_to_plot = [2],
                                                  plot_pinhole = False)
    
                            # If you want to mix and match, it assumes they have the same scintillator
                            if mixnmatch:
                                Geometry1.plot2Dfilled(ax=ax, view = 'YZ', element_to_plot = [0],
                                                       plot_pinhole = False)
                            
                            for i in range(2):
                                if read_slit[i] or mixnmatch:
                                    if plot_strike_points:
                                        Smap[i].strike_points.scatter(ax=ax, per=0.1, 
                                                                      mar_params = mar_params[i])
                                
                                    if plot_strikemap:
                                        Smap[i].plot_real(ax=ax, marker_params=marker_params[i],
                                                          line_params=line_params[i], factor=100.0, labels=True)
                                
                                    if plot_orbits:
                                        if read_slit[i]:
                                            orb[i] = ss.sinpa.orbits(runID=runid[i])
                                        orb[i].plot2D(ax=ax,line_params={'color': 'r'}, kind=(2,),factor=100.0)
                                        #orb[i].plot2D(ax=ax,line_params={'color': 'b'}, kind=(0,),factor=100.0)
                                        #orb[i].plot2D(ax=ax,line_params={'color': 'k'}, kind=(9,),factor=100.0)
                        
                          
                            fig.show()
                            
                            if plot_3D:
                                Geometry.plot3Dfilled(element_to_plot = [0,2])
                                ax3D = plt.gca()
                                if mixnmatch:
                                    Geometry1.plot3Dfilled(element_to_plot = [0],ax=ax3D)
                    
                                for i in range(2):
                                    if read_slit[i] or mixnmatch:
                                        # plot in red the ones colliding with the scintillator
                                        orb[i].plot3D(line_params={'color': 'r'}, kind=(2,), ax=ax3D, factor=100.0)
                                        # plot in blue the ones colliding with the collimator
                                        #orb[i].plot3D(line_params={'color': 'b'}, kind=(0,), ax=ax3D, factor=100.0)
                                        # plot the 'wrong orbits', to be sure that the are just orbits exceeding
                                        # the scintillator and nothing is wrong with the code
                                        #orb[i].plot3D(line_params={'color': 'k'}, kind=(9,), ax=ax3D, factor=100.0)
                            
                        for i in range(2):
                            if read_slit[i]:
                                try:
                                    Smap[i].calculate_resolutions(min_statistics = 100)
                                except:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                    min_gyro_res[i].append(np.nan)
                                    max_gyro_res[i].append(np.nan)
                                    pitch_p0_res[i].append(np.nan)
                                    gyro_1_res[i].append(np.nan)
                                    gyro_2_res[i].append(np.nan)
                                    gyro_3_res[i].append(np.nan)
                                    min_pitch_res[i].append(np.nan)
                                    max_pitch_res[i].append(np.nan)
                                    avg_collimating_factor[i].append(np.nan)
                                    pitch_p0_gyro_1_collimating_factor[i].append(np.nan)
                                    print('Could not calculate resolutions for slit ' + str(i+1))
                                    continue
    
                                if len(Smap[i].gyroradius)==0:
                                    min_gyro[i].append(np.nan)
                                    max_gyro[i].append(np.nan)
                                else:
                                    min_gyro[i].append(np.nanmin(Smap[i].gyroradius))
                                    max_gyro[i].append(np.nanmax(Smap[i].gyroradius))
    
                                index_p0 = np.where(Smap[i].pitch == p0[i])[0]
                                if len(index_p0) == 0:
                                    min_gyro_p0[i].append(np.nan)
                                    max_gyro_p0[i].append(np.nan)
                                else:
                                    min_gyro_p0[i].append(np.nanmin(Smap[i].gyroradius[index_p0]))
                                    max_gyro_p0[i].append(np.nanmax(Smap[i].gyroradius[index_p0]))
                            
                                if len(Smap[i].pitch)==0:
                                    min_pitch[i].append(np.nan)
                                    max_pitch[i].append(np.nan)
                                else:
                                    min_pitch[i].append(np.nanmin(Smap[i].pitch))
                                    max_pitch[i].append(np.nanmax(Smap[i].pitch)) 
                        
                                min_gyro_res[i].append(np.nanmin(Smap[i].resolution['Gyroradius']['sigma']))
                                max_gyro_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma']))
                            
                                idx_p0 = np.argmin( abs(Smap[i].strike_points.header['pitch'] - p0[i]))
                                try:
                                    pitch_p0_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx_p0]))
                                except:
                                    pitch_p0_res[i].append(np.nan)
                                
                                idx1 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -0.5))
                                idx2 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -1.0))
                                idx3 = np.argmin( abs(Smap[i].strike_points.header['gyroradius'] -1.5))
                                gyro_1_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx1,:]))
                                gyro_2_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx2,:]))
                                gyro_3_res[i].append(np.nanmax(Smap[i].resolution['Gyroradius']['sigma'][idx3,:]))
                                
                                min_pitch_res[i].append(np.nanmin(Smap[i].resolution['Pitch']['sigma']))
                                max_pitch_res[i].append(np.nanmax(Smap[i].resolution['Pitch']['sigma']))
                    
                                avg_collimating_factor[i].append(np.nanmean(Smap[i].collimator_factor_matrix))
                    
                                try:
                                    pitch_p0_gyro_1_collimating_factor[i].append(Smap[i].collimator_factor_matrix[idx2, idx_p0])
                                except:
                                    pitch_p0_gyro_1_collimating_factor[i].append(0)                    
                        
                                # for best case: haven't added a way to title these by slit yet
                                scan_string = scan_Parameters[scan_parameter]['scan_str']
                                diff = abs(best_param_value-value)/best_param_value
                                if scan_string == best_param and diff < 0.01:
                                    if plot_resolutions:
                                        Smap[i].plot_resolutions()
    
                                    if plot_gyro_res:
                                        Smap[i].plot_resolution_fits(var='Gyroradius',pitch=p0[i])
                                        plt.gcf().show()
    
                                    if plot_pitch_res:
                                        Smap[i].plot_resolution_fits(var='Pitch',gyroradius = 1.)
                                        plt.gcf().show()
    
                                    if plot_collimating_factor:
                                        Smap[i].plot_collimator_factor()
                                        plt.gcf().show()
                                        
                                    if plot_synthetic_signal:
                                        # I haven't tested this yet
                                        dist_file = '/afs/ipp/home/a/ajvv/ascot5/RUNS/W7X_distributions/250mm_FILD_distro.pck'
                                        distro = pickle.load( open( dist_file, "rb" ) )
        
                                        output = ss.fildsim.synthetic_signal_remap(distro, Smap[i],
                                                                                   rmin=0.1, rmax=4.0, dr=0.01,
                                                                                   pmin=55.0, pmax=105.0,
                                                                                   dp=1.0)
        
    
    
        
                                        fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                                                   facecolor='w', edgecolor='k', dpi=100)   
                                        ax_syn.set_xlabel('Pitch [$\\degree$]')
                                        ax_syn.set_ylabel('Gyroradius [cm]')
                                        ax_syn.set_ylim([0, 2.5])
                                        ss.fildsim.plot_synthetic_signal(output['gyroradius'], output['pitch']
                                                                   , output['signal'], 
                                                                   ax=ax_syn, fig=fig)
                                        if read_slit[0] and read_slit[1]:
                                            fig.title('Slit #' + str(i+1))
                                        fig.tight_layout()
                                        fig.show()
                                        
                        
                        del Smap
                                
                    ##plot metrics
                    if plot_metrics:
                        for i in range(2):
                            if read_slit[i]:
                            
                                x = scan_Parameters[scan_parameter]['scan_values']
                        
                                fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 12),
                                                   facecolor='w', edgecolor='k', dpi=100)
                
                                ax_gyro, ax_pitch, ax_gyro_res, ax_pitch_res = axarr[0,0], axarr[0,1], axarr[1,0], axarr[1,1]
                                ax_gyro.set_xlabel(scan_parameter + ' [cm]')
                                ax_gyro.set_ylabel('Gyroradius [cm]')
                
                                ax_gyro_res.set_xlabel(scan_parameter + ' [cm]')
                                ax_gyro_res.set_ylabel('Gyroradius resolution [cm]')
                    
                                ax_pitch_res.set_xlabel(scan_parameter + ' [cm]')
                                ax_pitch_res.set_ylabel('Pitch angle resolution [$\\degree$]')
                
                                ax_gyro.plot(x, min_gyro[i], marker = 'o', label = 'min')
                                ax_gyro.plot(x, max_gyro[i], marker = 'o', label = 'max')
                                ax_gyro.plot(x, min_gyro_p0[i], marker = 'o', label = 'min at ' + str(p0[i]) + '$\\degree$')
                                ax_gyro.plot(x, max_gyro_p0[i], marker = 'o', label = 'max at ' + str(p0[i]) + '$\\degree$')
                                ax_gyro.legend(loc='upper right')
                
                                ax_gyro_res.plot(x, gyro_1_res[i], marker = 'o', label = '0.5 cm')
                                ax_gyro_res.plot(x, gyro_2_res[i], marker = 'o', label = '1 cm')
                                ax_gyro_res.plot(x, gyro_3_res[i], marker = 'o', label = '1.5 cm')
                                ax_gyro_res.legend(loc='upper right')
                
                                ax_pitch_res.plot(x, min_pitch_res[i], marker = 'o', label = 'min')
                                ax_pitch_res.plot(x, max_pitch_res[i], marker = 'o', label = 'max')
                                ax_pitch_res.plot(x, pitch_p0_res[i], marker = 'o', label = str(p0[i]) + '$\\degree$')
                                ax_pitch_res.legend(loc='upper right')
    
                
                                ax_coll = axarr[0,1]
                                ax_coll.plot(x, avg_collimating_factor[i], marker = 'o', label='avg')
                                ax_coll.plot(x, pitch_p0_gyro_1_collimating_factor[i], marker = 'o', label='1 cm, ' + str(p0[i]) + '$\\degree$')
                                ax_coll.legend(loc='upper right')
                                ax_coll.set_xlabel(scan_parameter + ' [cm]')
                                ax_coll.set_ylabel('Average collimator factor %')
                
                                ax_gyro_res.set_ylim([0, 1.0])
                                ax_pitch_res.set_ylim([0, 5.0])
                            
                                if read_slit[0] and read_slit[1]:
                                    fig.suptitle('Slit '  + str(i+1) + '           ')
                                fig.tight_layout()
                                fig.show()                        
                    
                        
                        
    # -----------------------------------------------------------------------------
    # --- Section 3: Analyse the synthetic signal
    # -----------------------------------------------------------------------------
    smap = Smap[0]


    if plot_gyro_res:
        smap.plot_gyroradius_histograms(pitch = 85)
    
    if plot_pitch_res:
        smap.plot_pitch_histograms(gyroradius = 1.)

    if plot_collimating_factor:
        smap.plot_collimator_factor()
        plt.gcf().show()
    
    if plot_synthetic_signal:
        dist_file = '/afs/ipp/home/a/ajvv/ascot5/RUNS/W7X_distributions/pos_02_FILD_distro.pck'
        dist_file = '/afs/ipp/home/a/ajvv/ascot5/RUNS/W7X_distributions/250mm_FILD_distro.pck'
        distro = pickle.load( open( dist_file, "rb" ) )
        output = ss.SimulationCodes.FILDSIM.forwardModelling.synthetic_signal_remap(distro, smap,
                                                                                 rmin=0.75, 
                                                                                 rmax=3, 
                                                                                 dr=0.1,
                                                                                 pmin=55,
                                                                                 pmax=105,
                                                                                 dp=1.0)
        
        g_array, p_array, signal = output['gyroradius'], output['pitch'], output['signal']

        fig, ax_syn = plt.subplots(nrows=1, ncols=1, figsize=(6, 10),
                                   facecolor='w', edgecolor='k', dpi=100)   
        ax_syn.set_xlabel('Pitch [$\\degree$]')
        ax_syn.set_ylabel('Gyroradius [cm]')
        ax_syn.set_ylim([0, 2.5])
        
        cmap = cm.inferno # Colormap
        cmap._init()
        
        ss.SimulationCodes.FILDSIM.forwardModelling.plot_synthetic_signal(g_array, p_array
                                                           , signal, 
                                                           ax=ax_syn, fig=fig,
                                                           cmap = cmap)
        fig.tight_layout()
        fig.show()                        
                        
                        
                        
                        
