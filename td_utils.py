import warnings

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.ndimage import convolve
from scipy.stats import gaussian_kde
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from logzero import logger, loglevel

try:
    #from fastkde import fastKDE
    from scipy.stats.kde import gaussian_kde
except ModuleNotFoundError or ImportError:
    logger.warn('Failed to import FastKDE or/and gaussian_kde. Check your environment.')
except Exception as e:
    logger.warn('Unhandled exception when importing fastKDE and/or gaussian_kde: %s' % e)

from code_tools import check_value, check_dim
from td_imtools import image_curvature, signed_distance

## only method = sklearn-no-cv
def _compute_densities(image: np.ndarray, inside, narrowband,
                       n_samples: int,
                       kernel: str,
                       percentiles_range=(0, 100),
                       verbose: int = 30,
                       method: str = "sklearn"):
    check_value(method, ["sklearn", "sklearn-no-cv", "scipy-stats", "fastKDE"])

    loglevel(verbose)
    
    if image.ndim != 4:
        raise ValueError("You need to pass a 3 dim array. Got %d." % image.ndim)

    if image.shape[-1] > 4:
        logger.warn("You're in high dimension! You'll need more points and a LOT of computational power. Consider "
                    "using a dimensionality reduction technique, such as PCA.")

    if not np.any(inside) or not np.any(narrowband):
        raise ValueError("There are no samples in your inside or your narrowband.")

    n_depths, n_rows, n_cols, n_features = image.shape

    p_start, p_end = percentiles_range

    # values should have shape (# inside, n_features)
    values_inside = image[inside]
    values_narrowband = image[narrowband]

    assert values_inside.ndim == 2 and values_narrowband.ndim == 2, "Values do not have the right dimension."

    # percentiles is array of shape (n_features, 2)
    percentiles_inside = np.percentile(values_inside, [p_start, p_end], axis=0).T
    percentiles_narrowband = np.percentile(values_narrowband, [p_start, p_end], axis=0).T

    # Computing the values range (n_features, 2)
    values_range = np.zeros((n_features, 2))
    for i in range(n_features):
        values_range[i, 0] = np.min([percentiles_inside[i, 0], percentiles_narrowband[i, 0]])
        values_range[i, 1] = np.max([percentiles_inside[i, 1], percentiles_narrowband[i, 1]])

    # Computing the grid
    axis = []
    for i in range(n_features):
        axis_i = np.linspace(values_range[i, 0], values_range[i, 1], n_samples)
        axis.append(axis_i)
    grid = np.meshgrid(*axis)
    points_to_evaluate = np.array(grid).reshape(n_features, n_samples ** n_features).T

    # Just to be sure at first, but I checked and it's always true
    assert np.any(points_to_evaluate[0] != points_to_evaluate[1]), "Something went wrong when computing the features " \
                                                                   "array."

    if method == "sklearn-no-cv":
        std_inside = 1.06 * np.std(values_inside) * values_inside.shape[0] ** (-1 / 5)
        std_narrowband = 1.06 * np.std(values_narrowband) * values_inside.shape[0] ** (-1 / 5)

        kde_inside = KernelDensity(bandwidth=std_inside, kernel=kernel)
        kde_narrowband = KernelDensity(bandwidth=std_narrowband, kernel=kernel)

        kde_inside.fit(values_inside)
        kde_narrowband.fit(values_narrowband)

        pdf_inside = np.exp(kde_inside.score_samples(points_to_evaluate))
        pdf_narrowband = np.exp(kde_narrowband.score_samples(points_to_evaluate))

    else:
        raise ValueError

    # Normalizing the densities
    pdf_inside /= np.sum(pdf_inside)
    pdf_narrowband /= np.sum(pdf_narrowband)

    return {"pdf_inside": pdf_inside,
            "pdf_narrowband": pdf_narrowband,
            "kde_inside": kde_inside,
            "kde_narrowband": kde_narrowband,
            "std_inside": std_inside,
            "std_narrowband": std_narrowband,
            "z_space": points_to_evaluate}



## only bd_type = "neighborhood"
def _compute_boundaries(phi: np.ndarray, boundaries_type: str,
                        threshold_phi: float = None, neighborhood_size: int = None):
    
    if boundaries_type == "neighborhood":
        if neighborhood_size is None:
            raise ValueError("If you want to compute the boundaries with the neighborhood of the <0 part of the level "
                             "set, you need to specify a neighborhood_size. I got None. I suggest 2.")
        elif neighborhood_size % 2 == 0:
            raise ValueError("You need to specify an ODD number for the neighborhood size.")
        else:
            # change filter to 3d
            filter_ = np.ones((neighborhood_size, neighborhood_size,neighborhood_size), dtype=bool)
            filter_[neighborhood_size // 2, neighborhood_size // 2, neighborhood_size // 2] = 0
            inside = phi < 0
            narrowband_ext = convolve(inside, filter_, mode="constant") * (~ inside)
            narrowband_ext = narrowband_ext.astype(bool)
            narrowband = narrowband_ext  
            narrowband = narrowband.astype(bool)
    else:
        raise ValueError("Possible boundaries_type are : level-set | neighborhood | no-narrowband")

    if np.sum(inside) <= 0 or np.sum(narrowband) <= 0:
        logger.warning("Careful, the inside or narrowband is void.")
    return inside, narrowband


def dirac_eps(delta):
    """
A function to get a dirac approximation.
    d(x) = 1/(2 delta) * (1 + cos(pi * x / delta)) if x in [+- delta], 0 otherwise.
    Parameters
    ----------
    delta : float
        The bandwidth of the dirac function.
    """
    def discrete_dirac(x):
        res = 1 / (2 * delta) * (1 + np.cos(np.pi * x / delta))
        res[np.logical_or(x < -delta, x > delta)] = 0
        return res

    return discrete_dirac



def a_priori_estimates(phi: np.ndarray, dirac_threshold):
    mask = np.logical_and(phi > -dirac_threshold, phi < dirac_threshold)
    phi_values = phi[mask]
    n_inside = int(np.sum(phi_values < 0))
    n_outside = phi_values.size - n_inside
    return n_inside, n_outside

# only energy_type = bhatt
def _compute_energy(energy_type: str, pdf_inside, pdf_narrowband, eps=1e-10):

    if energy_type not in ["bhatt", "sandhu"]:
        raise ValueError("Energy must be either 'bhatt' or 'sabdhu'. Not %s." % energy_type)
    if pdf_inside.ndim != 1 or pdf_narrowband.ndim != 1:
        raise ValueError("You must provide 1-d array!")
    if pdf_inside.shape != pdf_narrowband.shape:
        raise ValueError("Densities must have the same shape!")
    if np.round(np.sum(pdf_inside), 5) != 1 or np.round(np.sum(pdf_narrowband), 5) != 1:
        logger.warning("Densities should be normalized!")

    if energy_type == "bhatt":
        b = np.sum(np.sqrt(pdf_inside * pdf_narrowband))
        b = eps if b == 0 else b
        return b
    
    
## only energy_type = bhatt， boundary_type = 
def _compute_image_energy_derivate(energy_type: str,
                                   I: np.ndarray, phi: np.ndarray,
                                   energy: float, z_space: np.ndarray,
                                   pdf_in: np.ndarray, pdf_out: np.ndarray,
                                   size_in: int, size_out: int,
                                   std_in: np.ndarray, std_out: np.ndarray,
                                   boundary_type: str,
                                   kernel: str,
                                   delta: float = None,
                                   narrowband: np.ndarray = None
                                   ):
    check_value(boundary_type, ["dirac", "narrowband", "no-narrowband"])
    use_dirac = boundary_type == "dirac"
    check_value(energy_type, ["bhatt", "sandhu"])
    bhatt_flow = energy_type == "bhatt"

    if use_dirac and delta is None:
        raise ValueError("If you want to use the Dirac to compute the derivative, you must specify an eps. "
                         "Got None.")
    elif not use_dirac:
        if narrowband is None:
            raise ValueError("If you want to use compute the derivative using the narrowband, you must specify one. "
                             "Got None.")
        elif narrowband.shape != I.shape[:-1]:
            raise ValueError("Narrowband and image do not have the same shape. Got %s and %s."
                             % (narrowband.shape, I.shape))

    if I.ndim != 4 or z_space.ndim != 2:
        raise ValueError("Image should be a 3-dim array and the z_space 2 dim array.")
    if I.shape[-1] != z_space.shape[-1]:
        raise ValueError("Image and z space must have the same number of dimensions.")
    if z_space.shape[0] != pdf_in.shape[0] or pdf_in.shape != pdf_out.shape:
        raise ValueError("Your z_space must have the same # of samples than your pdf_in and pdf_out arrays.")

    dirac = dirac_eps(delta)
    kernel_in = my_kernel(std_in, kernel_type=kernel)
    kernel_out = my_kernel(std_out, kernel_type=kernel)

    # To avoid division by 0. Not to be confused with delta, which is the threshold for the level set.
    epsilon = 1e-8
    n_depths, n_rows, n_cols = phi.shape

    # n_pixels will be the number of pixels to check
    mask = np.logical_and(phi > - delta, phi < delta) if use_dirac else narrowband
    # OUTPUT: n_pixels x n_features
    pixels_to_check = I[mask]
    # OUTPUT: n_pixels
    phi_to_check = phi[mask]
    n_pixels = pixels_to_check.shape[0]
    # n_z_samples the number of z samples
    n_z_samples, n_features = z_space.shape

    if n_pixels <= 10 and use_dirac:
        logger.warning("Too few samples will be computed for phi at this stage. You used:"
                       "\nDIRAC = True"
                       "\neps = %f"
                       "\nBut this creates a narrowband of size %d. Consider increasing eps."
                       % (delta, n_pixels))

    if bhatt_flow:
        # INPUT : n_z_samples / n_z_samples
        # OUTPUT: n_z_samples
        pdf_in_out_ratio = np.sqrt(pdf_in / (pdf_out + epsilon))
        pdf_out_in_ratio = np.sqrt(pdf_out / (pdf_in + epsilon))
        # INPUT : 1
        # OUTPUT: 1
        mean_shift = 1 / 2 * energy * (1 / size_in - 1 / size_out)
        size_of_array = max([z_space.dtype.itemsize, pixels_to_check.dtype.itemsize]) * n_pixels * n_z_samples * n_features
        # if it would take more than 10Go to compute this array:
        # TODO : there's still the n_pixels * n_z_samples memory constraint. I should split the pixel in batches too.
        if size_of_array > 1e10:
            n_split = int(size_of_array / 1e10)
            delta_split = int(n_pixels / n_split)
            norm_of_difference = np.zeros((n_pixels, n_z_samples))
            logger.debug("Array is too big (%3.2e o). Will split the computation in %d split."
                         % (size_of_array, n_split))
            for i in range(n_split):
                difference_z_I = (z_space - pixels_to_check[i*delta_split:(i+1)*delta_split]
                                  .reshape(delta_split, 1, n_features))
                norm_of_difference[i*delta_split:(i+1)*delta_split] = norm(difference_z_I, axis=-1)
            difference_z_I = (z_space - pixels_to_check[n_split*delta_split:]
                                  .reshape(n_pixels - n_split*delta_split, 1, n_features))
            norm_of_difference[n_split*delta_split:] = norm(difference_z_I, axis=-1)
        else:
            # INPUT :               n_z_samples x n_features - n_pixels x 1 x n_features
            # OUTPUT: n_pixels    x n_z_samples x n_features
            difference_z_I = z_space - np.reshape(pixels_to_check, (n_pixels, 1, n_features))
            # INPUT : n_pixels x n_z_samples x n_features
            # OUTPUT: n_pixels x n_z_samples
            norm_of_difference = norm(difference_z_I, axis=-1)
        # INPUT : n_pixels x n_z_samples
        # OUTPUT: n_pixels x n_z_samples
        kernel_out_result = kernel_out(norm_of_difference)
        # INPUT : n_pixels x n_z_samples * 1 x n_z_samples
        # OUTPUT: n_pixels x n_z_samples
        inside_integer_out = kernel_out_result * pdf_in_out_ratio.reshape((1, n_z_samples))
        # INPUT : n_pixels x n_z_samples
        # OUTPUT: n_pixels
        likelihood_plus = np.mean(inside_integer_out, axis=-1) / 2 / size_out
        # Same now for the other term
        kernel_in_result = kernel_in(norm_of_difference)
        inside_integer_in = kernel_in_result * pdf_out_in_ratio.reshape((1, n_z_samples))
        likelihood_minus = np.mean(inside_integer_in, axis=-1) / 2 / size_in
        # Finally :
        result = mean_shift + likelihood_plus - likelihood_minus

    else:
        # INPUT : n_z_samples / n_z_samples
        # OUTPUT: n_z_samples
        b_coeff = np.log(pdf_in / (pdf_out + epsilon) + epsilon)
        # INPUT :               n_z_samples x n_features - n_pixels x 1 x n_features
        # OUTPUT: n_pixels    x n_z_samples x n_features
        difference_z_I = z_space - pixels_to_check.reshape((n_pixels, 1, n_features))
        # INPUT : n_pixels x n_z_samples x n_features
        # OUTPUT: n_pixels x n_z_samples
        norm_of_difference = norm(difference_z_I, axis=-1)
        # INPUT : n_pixels x n_z_samples
        # OUTPUT: n_pixels x n_z_samples
        kernel_in_result = kernel_in(norm_of_difference)
        # INPUT : 1 - n_pixels x n_z_samples / n_z_samples
        # OUTPUT: n_pixels x n_z_samples
        g_in = 1 / size_in * (1 - kernel_in_result / (pdf_in + epsilon))
        # Same for the out:
        kernel_out_result = kernel_out(norm_of_difference)
        g_out = 1 / size_out * (1 - kernel_out_result / (pdf_out + epsilon))

        result = np.mean(b_coeff * (g_in + g_out), axis=-1) - np.mean(b_coeff) * np.mean(g_in + g_out, axis=-1)
        result /= energy

    if use_dirac:
        # INPUT : n_pixels - dirac(n_pixels) = n_pixels - n_pixels
        # OUTPUT: n_pixels
        result = result * dirac(phi_to_check)

    derivate = np.zeros((n_depths,n_rows, n_cols))
    derivate[mask] = result

    derivate /= np.max([derivate.max(), - derivate.min()])

    return derivate

    

def _compute_curvature(phi, eps, narrowband):
    """
Compute the curvature of the level-set. See image_curvature.
    """
    # if append newaxis to phi here, then have to run @handle4dim 
    curvature = image_curvature(phi[..., np.newaxis], chosen_loglevel=30)[..., 0]
    dirac = dirac_eps(eps)
    result = dirac(phi) * curvature

    return result


def _probability_initialization(image, z_space, pdf_in, pdf_out, phi, dirac_threshold):
    """
Use Bayes formula to compute the probability of being inside and being outside for every pixels. Then takes the
signed distance to 0.5 of the probability of being outside. Good way to reinitialize the level-sets.
    Parameters
    ----------
    z_space
        The feature space. 2-dim array. Shape: n_samples_in_z, n_features
    -------
Signed distance array.
    """
    n_in, n_out = a_priori_estimates(phi=phi, dirac_threshold=dirac_threshold)
    # print('pdf_out has shape %s and pdf_in has shape %s' % (pdf_out.shape, pdf_in.shape, ))
    p_in, p_out = predict_image(image=image, z_space=z_space, pdf_in=pdf_in, pdf_out=pdf_out, n_in=n_in, n_out=n_out)
    result = signed_distance(p_out, threshold=.5)
    return result


def predict_image(image: np.ndarray, z_space: np.ndarray, pdf_in: np.ndarray, pdf_out: np.ndarray,
                  n_in: int, n_out: int):
    """
Computes the probability of being inside and outside for every pixel, inverting the pdf estimate using bayes formula.
    n_in
        The number of samples inside
    n_out
        The number of samples outside
    Returns
    -------
Normalized probability of being inside, normalized probability of being outside.
    """
    check_dim(image, 4)
    check_dim(z_space, 2)
    check_dim(pdf_in, 1)
    check_dim(pdf_out, 1)
    if z_space.shape[0] != pdf_in.shape[0] or pdf_in.shape[0] != pdf_out.shape[0]:
        raise ValueError("The input parameters for the densities do not have the same size. Got: z_space, pdf_in,"
                         "pdf_out = %d, %d, %d" % (z_space.shape[0], pdf_in.shape[0], pdf_out.shape[0]))

    n_depths, n_rows, n_cols, n_features = image.shape
    n_pixels = n_rows * n_cols * n_depths
    n_z_samples = z_space.shape[0]
    eps = 1e-5
    # We have :
    # image   : n_pixels x      1      x n_features
    # z_space :            n_z_samples x n_features
    # result  : n_pixels x n_z_samples x n_features
    difference_array = image.reshape((n_pixels, 1, n_features)) - z_space.reshape((n_z_samples, n_features))
    # input   : n_pixels x n_z_samples x n_features
    # output  : n_pixels x n_z_samples
    distance_array = norm(difference_array, axis=-1)
    # We look at the closest sample in the z space
    # input   : n_pixels x n_z_samples
    # output  : n_pixels
    indices_min = np.argmin(distance_array, axis=-1)
    # We look at the estimated pdf in and out
    proba_in = pdf_in[indices_min].reshape(image.shape[:3])
    proba_out = pdf_out[indices_min].reshape(image.shape[:3])
    # We normalized these probabilities. To do that, we need to invert them, and thus we weight by the a priori
    # estimates of being in or out.
    proba_in_normalized = proba_in * n_in / (proba_in * n_in + proba_out * n_out)
    proba_out_normalized = proba_out * n_out / (proba_in * n_in + proba_out * n_out)
    return proba_in_normalized, proba_out_normalized


def my_kernel(std, kernel_type):


    check_value(kernel_type, ["gaussian", "epanechnikov"])

    if type(std) is np.ndarray:
        if std.ndim != 1:
            raise ValueError("You must provide a 1d array of std. Got %d." % std.ndim)
    else:
        std = np.array(std)
        std = std[np.newaxis]
    for sigma in std:
        if sigma <= 0:
            raise ValueError("You need to pass a positive float as std.")

    std.astype(np.float32)

    if kernel_type == "gaussian":
        def gaussian_kernel_func(x):
            return np.power(2 * np.pi, -1 / 2) * np.power(std, -1.) * np.exp(- np.power(x / std, 2.) / 2)
        return np.vectorize(gaussian_kernel_func, otypes=[np.float64])

    elif kernel_type == "epanechnikov": 
        def epanechnikov_kernel_func(x):
            # int_-1^+1 (1 - x**2) = 4/3
            res = 3 / (4 * std) * (1 - np.power(x / std, 2))
            res[res < 0] = 0
            return res
        return epanechnikov_kernel_func
