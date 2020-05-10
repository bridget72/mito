import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from logzero import loglevel, logger
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt
from skimage import filters as filters
from skimage.draw import circle
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import trange
import functools
import random
import raster_geometry as rst

from code_tools import check_value, check_dim

GRAD_X_FILTER = np.array([[[0, 0, 0],[1, 0,-1],[0, 0, 0]],
                          [[0, 0, 0],[1, 0,-1],[0, 0, 0]],
                          [[0, 0, 0],[1, 0,-1],[0, 0, 0]]])/2
GRAD_Y_FILTER = np.array([[[0, 1, 0],[0, 0, 0],[0,-1, 0]],
                          [[0, 1, 0],[0, 0, 0],[0,-1, 0]],
                          [[0, 1, 0],[0, 0, 0],[0,-1, 0]]])/2
GRAD_Z_FILTER = np.array([[[0, 0, 0],[1, 1, 1],[0, 0, 0]],
                          [[0, 0, 0],[0, 0, 0],[0, 0, 0]],
                          [[0, 0, 0],[-1,-1,-1],[0,0, 0]]])/2

def _x_derivative(array):
    return convolve(array, GRAD_X_FILTER, mode='reflect')

def _y_derivative(array):
    return convolve(array, GRAD_Y_FILTER, mode='reflect')

def _z_derivative(array):
    return convolve(array, GRAD_Z_FILTER, mode='reflect')

def _norm_derivative(array):
    dx = _x_derivative(array)
    dy = _y_derivative(array)
    dz = _z_derivative(array)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def signed_distance(phi: np.ndarray, threshold: float = 0):
    return distance_transform_edt(phi > threshold) - distance_transform_edt(phi < threshold)


def single_ball(shape,radius,position):
    arr = rst.sphere(shape,radius)
    arr = arr.astype(int)
    x = []
    y = []
    z = []
    center = int((shape+1)/2)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                if arr[i,j,k] == 1:
                    x.append(i-center+position[0])
                    y.append(j-center+position[1])
                    z.append(k-center+position[2])
    return x,y,z

def ball_init(r: float or list, c: float or list, d:float or list, radius: float or list, shape):
    if type(r) is not list:
        r = [r]
    if type(c) is not list:
        c = [c]
    if type(d) is not list:
        d = [d]
    if type(radius) is not list:
        radius = [radius]
    assert len(r) == len(c) == len(d) == len(radius)
    output = np.ones(shape)
    
    for (r_, c_, d_, radius_) in [(r[i], c[i], d[i], radius[i]) for i in range(len(r))]:
        x, y, z = single_ball(2*radius_-1,radius_,(r_,c_,d_))
        try:
            output[x,y,z] = -1
        except IndexError:
            print("")
        pass
    return signed_distance(output)


# suggest: raidus + 1 = 2 * step_size
def dense_initialization(step_size: float, radius: float or list, shape):
    x_list = []
    y_list = []
    z_list = []
    r_list = []
    r_num = int(shape[0]/step_size)
    c_num = int(shape[1]/step_size)
    d_num = int(shape[2]/step_size)
    for i in range(r_num):
        for j in range(c_num):
            for k in range(d_num):
                x_list.append(step_size*i+random.randint(0,radius-1))
                y_list.append(step_size*j+random.randint(0,radius-1))
                z_list.append(step_size*k+random.randint(0,radius-1))
                r_list.append(radius)
    result = ball_init(x_list,y_list,z_list,r_list,shape)
    return result

def handle_3_dim(name):
    def handle3_without_name(func):
        @functools.wraps(func)
        def handle3_decorator(image, *args, chosen_loglevel=None, **kwargs):
            check_dim(image, [3,4])

            if image.ndim == 3:
                wrapper_returns = func(image, *args, **kwargs)

            else:
                previous_loglevel = logger.level
                if chosen_loglevel is None:
                    chosen_loglevel = logger.level
                loglevel(chosen_loglevel)
                iteration = trange if chosen_loglevel <= 20 else range

                t1 = time()
                n_depths,n_rows, n_cols, n_channels = image.shape
                wrapper_returns = np.zeros(image.shape)

                for d in iteration(n_channels):
                    wrapper_returns[..., d] = func(image[..., d], *args, **kwargs)

                logger.info("%s done in %5.1fs!" % (name, time() - t1))
                loglevel(previous_loglevel)

            return wrapper_returns
        return handle3_decorator
    return handle3_without_name

def local_max_initialization(image, min_distance, radius):    
    coor = peak_local_max(image, min_distance)
    x_list = coor[:,0]
    y_list = coor[:,1]
    z_list = coor[:,2]
    result = ball_init(x_list,y_list,z_list, radius, image.shape)
    return result

def local_max(image, min_distance, radius):
    x_list = []
    y_list = []
    z_list = []
    r_list = []    
    for i in range(image.shape[0]):
        coor = peak_local_max(image[i], min_distance)
        y = list(coor[:, 0])
        z = list(coor[:, 1])
        x_list.extend([i]*len(y)) 
        y_list.extend(y)
        z_list.extend(z)
        r_list.extend([radius]*len(y)) 
    result = ball_init(x_list,y_list,z_list,r_list,image.shape)
    return result

@handle_3_dim("Curvature")
def image_curvature(image: np.ndarray):
    """
Compute curvature of a 3d image array
    """
    dx = _x_derivative(image)
    dxx= _x_derivative(dx)
    dy = _y_derivative(image)
    dyy= _y_derivative(dy)
    dz = _z_derivative(image)
    dzz= _z_derivative(dz)
    dxy= _y_derivative(dx)  
    dxz= _z_derivative(dx) 
    dyz= _z_derivative(dy)
    nom1 = (dyy+dzz)*dx**2 + (dxx+dzz)*dy**2 + (dxx+dyy)*dz**2
    nom2 = 2 * (dx*dy*dxy + dx*dz*dxz + dy*dz*dyz)
    return np.power(dx ** 2 + dy ** 2 + dz ** 2+ 1e-10, -3 / 2) * (- nom1 + nom2)
