import numpy as np
import os

from skimage import io
from skimage.segmentation._chan_vese import _cv_small_disk
from skimage.segmentation._chan_vese import _cv_checkerboard
from skimage.draw import circle
from skimage.filters import gaussian

import matplotlib.pyplot as plt
from matplotlib import use as mpl_backend
from mpl_toolkits.mplot3d import Axes3D
 
mpl_backend('Qt5Agg')
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from logzero import logger, loglevel
from zipfile import BadZipFile
from tqdm import trange

from code_tools import check_value
from td_imtools import signed_distance, ball_init, dense_initialization, local_max_initialization, local_max
from td_utils import _compute_densities, _compute_boundaries, a_priori_estimates, \
    _compute_energy, _compute_image_energy_derivate, _compute_curvature, _probability_initialization, predict_image


class MichailovichFlow:
    phi: np.ndarray

    def __init__(self, image: np.ndarray,
                 energy_type: str,
                 narrowband_type: str,
                 alpha_curvature: float = 1,
                 narrowband_neighborhood=3,
                 narrowband_threshold: float = .1,
                 dirac_delta: float = .1,
                 n_samples: int = 50,
                 kernel: str = "gaussian",
                 initial_segmentation: np.ndarray or str = "circle"):
        
        check_value(narrowband_type, ["level-set", "neighborhood", "no-narrowband"])
        check_value(energy_type, ["bhatt", "sandhu"])
        check_value(kernel, ["gaussian", "epanechnikov"])

        if energy_type == "sandhu":
            raise NotImplementedError("Sandhu energy does not work for now.")

        self.kernel = kernel
        self.energy_type = energy_type
        self.narrowband_type = narrowband_type
        self.narrowband_neighborhood = narrowband_neighborhood
        self.dirac_threshold = dirac_delta
        self.n_samples = n_samples
        self.narrowband_threshold = narrowband_threshold
        self.alpha_curvature = alpha_curvature
                                                                                                  
        if image.dtype.type not in [np.float, np.float64, np.float32, np.float16]:
            raise ValueError("You must pass a float array. Got: %s." % image.dtype.type)
        if image.ndim == 3:
            self.I = image[..., np.newaxis]
        elif image.ndim == 4:
            self.I = image
        else:
            raise ValueError("You must pass a 3d or 4d array. Got %d." % image.ndim)

        self.n_features = self.I.shape[-1]

        for d in range(self.I.shape[-1]):                                                         
            self.I[..., d] = self.I[..., d] - self.I[..., d].min()
            self.I[..., d] = self.I[..., d] / self.I[..., d].max()
        #TODO: check if this init works in 3d 
        if type(initial_segmentation) is str:
            if initial_segmentation == "circle":
                initial_segmentation = _cv_small_disk(self.I.shape[:-1])
                initial_segmentation *= -1  # We want the <0 to be the inside
            elif initial_segmentation == "checkerboard":
                initial_segmentation = _cv_checkerboard(self.I.shape[:-1], square_size=5)
                initial_segmentation *= -1
        else:
            # noinspection PyUnresolvedReferences
            if type(initial_segmentation) != np.ndarray or initial_segmentation.shape != self.I.shape[:-1]:
                raise ValueError("The dimensions of initial level set do not match the dimensions of image.")
                
        # noinspection PyTypeChecker
        self.phi = signed_distance(initial_segmentation)
        self.energies = []
        self.curvatures = []
        self.image_energy_derivate = []
        self.insides = []
        self.narrowbands = []
        self.pdf_inside = []
        self.pdf_narrowband = []
        self.z_spaces = []
        self.phis = [self.phi.copy()]
        self.n_iterations = 0
        self.time = [.0]
        self.stds = []

    def phi_derivate(self, kde_method: str, boundary_type: str, compute_std: bool, verbose: int = 30):
        """
Compute the derivate of the level-set function.
        """
        inside, narrowband = _compute_boundaries(phi=self.phi, boundaries_type=self.narrowband_type,
                                                 threshold_phi=self.narrowband_threshold,
                                                 neighborhood_size=self.narrowband_neighborhood)
        size_inside, size_narrowband = a_priori_estimates(self.phi, self.dirac_threshold)

        if size_inside <= 5 or size_narrowband <= 5:
            logger.warning("There are too few samples in the inside or the narrowband.")
            return {"output_code": 0,
                    "output_message": "No samples in the inside or the narrowband. "
                                      "Inside : %d. Narrowband : %d." % (size_inside, size_narrowband)}

        if len(self.stds) != 0:
            std = self.stds[-1]
        else:
            std = None
        values = _compute_densities(self.I, inside, narrowband, n_samples=self.n_samples,
                                    method=kde_method, verbose=verbose, kernel=self.kernel)

        # pdf_inside is in the Z space!
        energy = _compute_energy(energy_type=self.energy_type,
                                 pdf_inside=values["pdf_inside"], pdf_narrowband=values["pdf_narrowband"])

        image_energy_derivate = _compute_image_energy_derivate(
            energy_type=self.energy_type,
            I=self.I, phi=self.phi, energy=energy, z_space=values["z_space"],
            pdf_in=values["pdf_inside"], pdf_out=values["pdf_narrowband"],
            size_in=size_inside, size_out=size_narrowband,
            std_in=values["std_inside"], std_out=values["std_narrowband"],
            boundary_type=boundary_type, delta=self.dirac_threshold, narrowband=narrowband,
            kernel=self.kernel)

        # And now, curvature
        curvature = _compute_curvature(phi=self.phi, eps=self.dirac_threshold, narrowband=narrowband)
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min())
        curvature *= - self.alpha_curvature

        return {"output_code": 1,
                "output_message": "all went well",
                "energy": energy,
                "narrowband": narrowband,
                "inside": inside,
                "pdf_inside": values["pdf_inside"],
                "pdf_outside": values["pdf_narrowband"],
                "std": (values["std_inside"], values["std_narrowband"]),
                "energy_curvature": curvature,
                "image_energy_derivate": image_energy_derivate,
                "full_derivate": image_energy_derivate + curvature,
                "z_space": values["z_space"]}

    def run_flow(self, dt: float, n_iteration: int, boundary_type: str, kde_method: str = "sklearn",
                 verbose: int = 30,
                 never_break_when_energy_increases: bool = False,
                 display=False,
                 signed_distance_reset_period: int = 100,
                 proba_estimate_reset_period: int = 1000,
                 save_debug: bool = True,
                 save_file: bool = False,
                 save_dir: str = None):
        
        loglevel(verbose)

        if save_file:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, "info.txt"), "a") as f:
                f.write(f'energy_type : {self.energy_type}'
                        f'\nn_iteration : {n_iteration}'
                        f'\ndt : {dt}'
                        f'\nn_samples : {self.n_samples}'
                        f'\nalpha_curvature : {self.alpha_curvature}'
                        f'\nn_features : {self.n_features}'
                        f'\nkernel : {self.kernel}'
                        f'\nboundary_type : {boundary_type}'
                        f'\nnarrowband_neighborhood : {self.narrowband_neighborhood}'
                        f'\ndirac_threshold : {self.dirac_threshold}'
                        )
            np.save(os.path.join(save_dir, "image.npy"), self.I)

        compute_std_at_next_step = True

        for step in trange(n_iteration):
            self.n_iterations += 1

            if step % signed_distance_reset_period == 0 and step != 0:
                self.phi = signed_distance(self.phi)

            derivate = self.phi_derivate(kde_method=kde_method, boundary_type=boundary_type,
                                         compute_std=compute_std_at_next_step, verbose=verbose)

            if derivate["output_code"] == 0:
                logger.warning("Got output code %d at step %d. Output message is : \n%s" %
                               (derivate["output_code"], step, derivate["output_message"]))
                logger.warning("Attempting to recover by resetting the level-sets to the signed-distance.")
                self.phi = signed_distance(self.phi)
                continue

            energy_derivate = derivate["full_derivate"]

            if step % proba_estimate_reset_period == 0 and step != 0:
                logger.debug('Predicting on the whole image')
                self.phi = _probability_initialization(image=self.I, z_space=derivate["z_space"],
                                                       pdf_in=derivate["pdf_inside"],
                                                       pdf_out=derivate["pdf_outside"],
                                                       phi=self.phi, dirac_threshold=self.dirac_threshold)
            else:
                self.phi -= dt * energy_derivate


            if save_file:
                np.savez_compressed("%s/%d.npz" % (save_dir, self.n_iterations),
                                    phi=self.phi,
                                    energy=derivate['energy'],
                                    curvatures=derivate["energy_curvature"],
                                    image_energy_derivate=derivate["image_energy_derivate"],
                                    insides=derivate["inside"],
                                    narrowbands=derivate["narrowband"],
                                    pdf_inside=derivate["pdf_inside"],
                                    pdf_narrowband=derivate["pdf_outside"],
                                    stds=derivate["std"],
                                    z_spaces=derivate["z_space"])

            if save_debug:
                self.phis.append(self.phi.copy())
                self.energies.append(derivate["energy"])
                self.curvatures.append(derivate["energy_curvature"])
                self.image_energy_derivate.append(derivate["image_energy_derivate"])
                self.insides.append(derivate["inside"])
                self.narrowbands.append(derivate["narrowband"])
                self.pdf_inside.append(derivate["pdf_inside"])
                self.pdf_narrowband.append(derivate["pdf_outside"])
                self.stds.append(derivate["std"])
                self.z_spaces.append(derivate["z_space"])
                self.time.append(self.time[-1] + dt)

            if self.n_iterations != 1 and not never_break_when_energy_increases:
                if self.energies[step] > self.energies[step - 1]:
                    logger.warning("The energy increased. Do you want to stop now?(y/n/N:Never)")
                    answer = ""
                    while answer not in ["y", "n", "N"]:
                        answer = input("y/n/N:")
                    if answer == "y":
                        break
                    elif answer == "N":
                        never_break_when_energy_increases = True

        if display: 
            ## display the final contour on slice positioned at 1/2 depth
            
            depth = (self.I[..., 0].shape[0])/2
            depth = int(depth)
            fig = plt.figure()
            fig.suptitle('Final contour overlaid on raw stack at depth %d' %depth)
            plt.imshow(self.I[..., 0][depth,:,:], cmap=plt.get_cmap("gray"), alpha=.7)
            plt.contour(self.phis[self.n_samples][depth,:,:],levels = [0], cmap=plt.get_cmap("Spectral"), alpha=.4)
            
            image_selector = Slider(ax=fig.add_axes((.1, .03, .8, .02)), label="Image index", valmin=0,
                                    valmax=self.n_iterations - 1, valinit=0, valfmt="%d", valstep=1)
            
            
            gs = GridSpec(2, 5)
            fig = plt.figure(figsize=(40, 10))
            axes = (fig.add_subplot(gs[0, 0]),
                    fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[0, 2]),
                    fig.add_subplot(gs[0, 3]),
                    fig.add_subplot(gs[0, 4]),
                    fig.add_subplot(gs[1, 1:-1], projection=None if self.n_features != 2 else "3d"))

            # TODO : put all that in a function
            #!!!!!!! 
            axes[0].imshow(self.I[..., 0][depth,:,:], cmap=plt.get_cmap("coolwarm"), alpha=.7)
            #!!!!!!showing result from step 50 
            inside = axes[0].imshow(self.phis[self.n_samples][depth,:,:] < 0, cmap=plt.get_cmap("gray"), alpha=.4)
            axes[0].set_title("Computed inside %d" % 0)

            axes[1].imshow(self.I[..., 0][depth,:,:], cmap=plt.get_cmap("coolwarm"), alpha=.7)
            narrowband = axes[1].imshow(self.narrowbands[0][depth,:,:], cmap=plt.get_cmap("gray"), alpha=.4)
            axes[1].set_title("Narrowbands %d" % 0)

            derivate = axes[2].imshow(self.image_energy_derivate[0][depth,:,:], cmap=plt.get_cmap("coolwarm"))
            axes[2].set_title("Image energy derivate %d" % 0)
            plt.colorbar(derivate, ax=axes[2], orientation="horizontal")

            curvature = axes[3].imshow(self.curvatures[0][depth,:,:], cmap=plt.get_cmap("coolwarm"))
            axes[3].set_title("Curvature %d" % 0)
            plt.colorbar(curvature, ax=axes[3], orientation="horizontal")

            phi = axes[4].imshow(self.phis[0][depth,:,:], cmap=plt.get_cmap("coolwarm"))
            axes[4].set_title("Level sets %d" % 0)
            plt.colorbar(phi, ax=axes[4], orientation="horizontal")

            if self.n_features == 1:
                density_inside, = axes[5].plot(self.z_spaces[0], self.pdf_inside[0], "g-", label="density inside")
                density_narrowband, = axes[5].plot(self.z_spaces[0], self.pdf_narrowband[0], "r-", label="density "
                                                                                                         "narrowband")
                axes[5].set_title("Image_e = %4.3e; std=%s" % (self.energies[0], self.stds[0]))
                axes[5].legend()
            elif self.n_features == 2:
                density_inside, = axes[5].plot(self.z_spaces[0][:, 0],
                                               self.z_spaces[0][:, 1],
                                               self.pdf_inside[0],
                                               linestyle="", c="g", marker="o",
                                               label="density inside")
                density_narrowband, = axes[5].plot(self.z_spaces[0][:, 0],
                                                   self.z_spaces[0][:, 1],
                                                   self.pdf_narrowband[0],
                                                   linestyle="", c="r", marker="^",
                                                   label="density narrowband")
                axes[5].set_xlabel("Feature 1")
                axes[5].set_ylabel("Feature 2")
                axes[5].set_zlabel("Density")

            image_selector = Slider(ax=fig.add_axes((.1, .03, .8, .02)), label="Image index", valmin=0,
                                    valmax=self.n_iterations - 1, valinit=0, valfmt="%d", valstep=1)

            fig.suptitle("ENERGY TYPE = {}"
                         "\nBOUNDARY TYPE = {}"
                         "\nBOUNDARY PARAM = {}"
                         "\nN_ITERATIONS = {:d}"
                         "\nKERNEL = {},"
                         "\nTIME = {:4.3f}"
                         .format(self.energy_type,
                                 boundary_type,
                                 self.narrowband_neighborhood if boundary_type == "narrowband" else
                                 self.dirac_threshold,
                                 self.n_iterations,
                                 self.kernel,
                                 dt),
                         x=0.05, y=.95)
            
            def update(index):
                index = int(index)
                inside.set_data(self.insides[index])
                axes[0].set_title("Computed inside %d" % index)
                narrowband.set_data(self.narrowbands[index])
                axes[1].set_title("Narrowbands %d" % index)
                derivate.set_data(self.image_energy_derivate[index])
                derivate.set_clim(self.image_energy_derivate[index].min(), self.image_energy_derivate[index].max())
                axes[2].set_title("Image energy derivate %d" % index)
                curvature.set_data(self.curvatures[index])
                derivate.set_clim(self.curvatures[index].min(), self.curvatures[index].max())
                axes[3].set_title("Curvature %d" % index)
                phi.set_data(self.phis[index])
                phi.set_clim(self.phis[index].min(), self.phis[index].max())
                axes[4].set_title("Level sets %d" % index)
                if self.n_features == 1:
                    density_inside.set_xdata(self.z_spaces[index])
                    density_inside.set_ydata(self.pdf_inside[index])
                    density_narrowband.set_xdata(self.z_spaces[index])
                    density_narrowband.set_ydata(self.pdf_narrowband[index])
                    axes[5].set_xlim(self.z_spaces[index][[0, -1]])
                    axes[5].set_ylim(np.min([self.pdf_inside[index].min(), self.pdf_narrowband[index].min()]),
                                     np.max([self.pdf_inside[index].max(), self.pdf_narrowband[index].max()]))
                elif self.n_features == 2:
                    density_inside.set_data(self.z_spaces[index][:, 0], self.z_spaces[index][:, 1])
                    density_inside.set_3d_properties(self.pdf_inside[index])
                    density_narrowband.set_data(self.z_spaces[index][:, 0], self.z_spaces[index][:, 1])
                    density_narrowband.set_3d_properties(self.pdf_narrowband[index])
                    axes[5].set_xlim(self.z_spaces[index][([0, -1], 0)])
                    axes[5].set_ylim(self.z_spaces[index][([0, -1], 1)])
                    axes[5].set_zlim(np.min([self.pdf_inside[index].min(), self.pdf_narrowband[index].min()]),
                                     np.max([self.pdf_inside[index].max(), self.pdf_narrowband[index].max()]))
                axes[5].set_title("Image_e = %4.3e; std=%s" % (self.energies[index], self.stds[index]))

            image_selector.on_changed(update)
            plt.show()
            return fig, axes, image_selector


def get_fig_axis(projection=None):
    """
simple function to get fig and axes in the right projection.

    Parameters
    ----------
    projection
        None or '3d'.

    Returns
    -------
fig, axes_dict usable in `show_flow` function.
    """
    gs = GridSpec(2, 5)
    fig = plt.figure(figsize=(40, 10))
    axes_dict = {'contours': {'ax': fig.add_subplot(gs[0, 0])},
                 'phi': {'ax': fig.add_subplot(gs[0, 3])},
                 'pdf': {'ax': fig.add_subplot(gs[1, 1:-1], projection=projection)}},
    return fig, axes_dict


def show_flow(path_to_dir, plot_to_ax: dict, fig, smoothing = 0):
    """
A function to show a flow with a beautiful slider, from a directory containing a lot of npz archives generated
through the run_flow method of MichailovichFlow class, with save_file set to True.

    Parameters
    ----------
    path_to_dir
        Path to the directory of npz files.
    plot_to_ax
        A dict of dict. Dict keys should be in: ['pdf', 'contours', 'image_energy_derivate', 'curvatures', 'phi']
        Each key should have value a dict, which should contain at least one 'ax' entry, containing an ax.
                       Example: {'pdf':{'ax': my_ax}, 'curvatures':{'ax': other_ax, 'cmap': my_cmap}
    fig
        A matplotlib figure
    smoothing
        Whether to smooth the level-set with gaussian blur.

    Returns
    -------
fig, ax_dict, reference to the slider -> to keep it usable and not be GC.
    """
    handled_plots = ['pdf', 'contours', 'image_energy_derivate', 'curvatures', 'phi']
    for k in plot_to_ax.keys():
        if k not in handled_plots:
            raise RuntimeError("This plot is not handled: %s. Try one among: %s" % (k, handled_plots))
        if type(plot_to_ax[k]) is not dict:
            raise RuntimeError("you need to pass dicts, but got: %s" % type(plot_to_ax[k]))
        if 'ax' not in plot_to_ax[k].keys():
            raise RuntimeError('Each dict entry must have an "ax" key. Found none.')
    if not os.path.isdir(path_to_dir):

        raise NotADirectoryError("The path you provided is not a directory : %s" % path_to_dir)
    try:
        with open(os.path.join(path_to_dir, "info.txt"), "r") as f:
            config = {}
            lines = f.readlines()
            for x in lines:
                current_line = x[:-1] if x.endswith('\n') else x
                key, value = current_line.split(" : ")
                config[key] = value
    except FileNotFoundError:
        logger.error("There is no 'info.txt' file in the provided directory.")
        raise
    except Exception as e:
        logger.error("Parsing the config file failed. Please check it that it has multiples lines, each separated by "
                     "a ':'. Exception : %s" % e)
        raise

    config['n_iteration'] = int(config['n_iteration'])
    config['dt'] = float(config['dt'])
    config['n_features'] = int(config['n_features'])

    if "pdf" in plot_to_ax.keys():
        if config['n_features'] == 2:
            assert 'proj' in plot_to_ax['pdf']['ax'].properties(), "The ax you passed for the pdf has no 3d " \
                                                               "projection."
        elif config['n_features'] == 1:
            assert 'proj' not in plot_to_ax['pdf']['ax'].properties(), "You passed a '3d' axis when there are 1d plot to make."

    image = np.load(os.path.join(path_to_dir, "image.npy"))

    values_to_plot = {'phi': [],
                      'narrowbands': [],
                      'insides': [],
                      'image_energy_derivate': [],
                      'curvatures': [],
                      'z_spaces': [],
                      'pdf_inside': [],
                      'pdf_narrowband': [],
                      'energy': [],
                      'stds': [],
                      }

    list_of_files = [int(x.split('.')[0]) for x in os.listdir(path_to_dir) if x.endswith(".npz")]
    n_iteration = len(list_of_files)
    if sorted(list_of_files) != list(range(1, n_iteration + 1)):
        raise FileNotFoundError("A regular time step seems not to be respected. This is not handled yet. Got: %s"
                                % list_of_files)

    logger.info("Loading the files in the provided directory.")
    for i in trange(1, n_iteration+1):
        try:
            current_step = np.load(os.path.join(path_to_dir, f'{i}.npz'))
        except BadZipFile:
            logger.warn('There were a problem loading time %d. Skipping it...' % i)
            current_step = previous_step
        except Exception as e:
            logger.error('Unhandled exception when unzipping time %d. \n %s' % (i, e))
            raise
        for k in values_to_plot.keys():
            try:
                values_to_plot[k].append(current_step[k])
            except KeyError:
                logger.error("%s is not in the archive at time %d" % (k, i))
                raise
            except Exception as e:
                logger.error("At time %d, when looking for %s, got exception : %s" % (i, k, e))
                raise
        previous_step = current_step
    previous_step.close()
    current_step.close()

    for k in plot_to_ax.keys():
        if k == 'pdf':
            if config['n_features'] == 1:
                density_inside, = plot_to_ax[k]['ax'].plot(values_to_plot['z_spaces'][0],
                                                             values_to_plot['pdf_inside'][0],
                                                             "g-", label="density inside")
                density_narrowband, = plot_to_ax[k]['ax'].plot(values_to_plot['z_spaces'][0],
                                                                 values_to_plot['pdf_narrowband'][0],
                                                                 "r-", label="density narrowband")
                plot_to_ax[k]['ax'].set_title("Image_e = %4.3e; std=%s"
                                                % (values_to_plot['energy'][0], values_to_plot['stds'][0]))
                plot_to_ax[k]['ax'].legend()
            elif config['n_features'] == 2:
                density_inside, = plot_to_ax[k]['ax'].plot(values_to_plot['z_spaces'][0][:, 0],
                                               values_to_plot['z_spaces'][0][:, 1],
                                               values_to_plot['pdf_inside'][0],
                                               linestyle="", c="g", marker="o",
                                               label="density inside")
                density_narrowband, = plot_to_ax[k]['ax'].plot(values_to_plot['z_spaces'][0][:, 0],
                                                   values_to_plot['z_spaces'][0][:, 1],
                                                   values_to_plot['pdf_narrowband'][0],
                                                   linestyle="", c="r", marker="^",
                                                   label="density narrowband")
                plot_to_ax[k]['ax'].set_xlabel("Feature 1")
                plot_to_ax[k]['ax'].set_ylabel("Feature 2")
                plot_to_ax[k]['ax'].set_zlabel("Density")
        elif k=='contours':
            plot_to_ax[k]['ax'].imshow(image[..., 0], cmap=plt.get_cmap("gray"), alpha=.7)
            if smoothing == 0:
                plot_to_ax[k]['ax'].contour(values_to_plot['phi'][0], [0], colors='r')
            else:
                plot_to_ax[k]['ax'].contour(gaussian(values_to_plot['phi'][0], sigma=smoothing), [0], colors='r')
            plot_to_ax[k]['ax'].set_title("Computed inside %d" % 0)
        else:
            plot_to_ax[k]['plot_var'] = plot_to_ax[k]['ax'].imshow(values_to_plot[k][0], **plot_to_ax[k])

    image_selector = Slider(ax=fig.add_axes((.1, .03, .8, .02)), label="Image index", valmin=0,
                            valmax=n_iteration - 1, valinit=0, valfmt="%d", valstep=1)

    fig.suptitle("ENERGY TYPE = {}"
                 "\nBOUNDARY TYPE = {}"
                 "\nBOUNDARY PARAM = {}"
                 "\nN_ITERATIONS = {}"
                 "\nKERNEL = {},"
                 "\nTIME = {:4.3f}"
                 .format(config['energy_type'],
                         config['boundary_type'],
                         config['narrowband_neighborhood'] if config['boundary_type'] == "narrowband" else
                         config['dirac_threshold'],
                         config['n_iteration'],
                         config['kernel'],
                         config['dt']),
                 x=0.05, y=.95)

    def update(index):
        index = int(index)
        for k in plot_to_ax.keys():
            if k == 'pdf':
                if config['n_features'] == 1:
                    density_inside.set_xdata(values_to_plot['z_spaces'][index])
                    density_inside.set_ydata(values_to_plot['pdf_inside'][index])
                    density_narrowband.set_xdata(values_to_plot['z_spaces'][index])
                    density_narrowband.set_ydata(values_to_plot['pdf_narrowband'][index])
                    plot_to_ax[k]['ax'].set_xlim(values_to_plot['z_spaces'][index][[0, -1]])
                    plot_to_ax[k]['ax'].set_ylim(
                        np.min([values_to_plot['pdf_inside'][index].min(), values_to_plot['pdf_narrowband'][
                            index].min()]),
                        np.max([values_to_plot['pdf_inside'][index].max(), values_to_plot['pdf_narrowband'][
                            index].max()]))
                elif config['n_features'] == 2:
                    density_inside.set_data(values_to_plot['z_spaces'][index][:, 0],
                                            values_to_plot['z_spaces'][index][:, 1])
                    density_inside.set_3d_properties(values_to_plot['pdf_inside'][index])
                    density_narrowband.set_data(values_to_plot['z_spaces'][index][:, 0],
                                                values_to_plot['z_spaces'][index][:, 1])
                    density_narrowband.set_3d_properties(values_to_plot['pdf_narrowband'][index])
                    plot_to_ax[k]['ax'].set_xlim(values_to_plot['z_spaces'][index][([0, -1], 0)])
                    plot_to_ax[k]['ax'].set_ylim(values_to_plot['z_spaces'][index][([0, -1], 1)])
                    plot_to_ax[k]['ax'].set_zlim(
                        np.min([values_to_plot['pdf_inside'][index].min(), values_to_plot['pdf_narrowband'][
                            index].min()]),
                        np.max([values_to_plot['pdf_inside'][index].max(), values_to_plot['pdf_narrowband'][
                            index].max()]))
                plot_to_ax[k]['ax'].set_title("Image_e = %4.3e; std=%s" % (values_to_plot['energy'][index],
                                                                           values_to_plot['stds'][index]))
            elif k == 'contours':
                # plot_to_ax[k]['ax'].clear()
                # plot_to_ax[k]['ax'].imshow(image[..., 0], cmap=plt.get_cmap("gray"), alpha=.7)
                try:
                    plot_to_ax[k]['ax'].collections[0].remove()
                except IndexError:
                    pass
                if smoothing == 0:
                    plot_to_ax[k]['ax'].contour(values_to_plot['phi'][index], [0], colors='r')
                else:
                    plot_to_ax[k]['ax'].contour(gaussian(values_to_plot['phi'][index],
                                                sigma=smoothing), [0], colors='r')
                plot_to_ax[k]['ax'].set_title("Computed inside %d" % index)
            else:
                plot_to_ax[k]['plot_var'].set_data(values_to_plot[k][index])
                plot_to_ax[k]['ax'].set_clim(values_to_plot[k][index].min(),
                                             values_to_plot[k][index].max())
                plot_to_ax[k]['ax'].set_title("%s, it %d" % (k, index))

    image_selector.on_changed(update)
    plt.show()
    return fig, plot_to_ax, image_selector


def show_prediction_on_all_image(image, z_spaces, pdfs_inside, pdfs_outside, phis, dirac_threshold):
    """
Function to show the evolution of the prediction on the all image, with nice slider to show the boundaries.

    Parameters
    ----------
    image
        The feature image
    z_spaces
        The Z_space points
    pdfs_inside
        The pdf inside
    pdfs_outside
        The pdf outside
    phis
        The list of level set function
    dirac_threshold
        The dirac parameter of the level-set.

    Returns
    -------
References to the sliders.
    """
    probas_in = []
    probas_out = []
    logistic_ratios = []
    n_iteration = len(z_spaces)

    for i in trange(n_iteration):
        n_in, n_out = a_priori_estimates(phis[i], dirac_threshold)
        p_in, p_out, log_ratio = predict_image(image, z_spaces[i], pdfs_inside[i], pdfs_outside[i], n_in, n_out)
        probas_in.append(p_in)
        probas_out.append(p_out)
        logistic_ratios.append(log_ratio)

    fig, axes = plt.subplots(1, 4)
    axes[0].imshow(image[..., 0])
    contour = [axes[0].contour(logistic_ratios[0], [.5], alpha=.5, colors='r')]
    axes[0].set_title("Raw")
    im_in = axes[1].imshow(probas_in[0].reshape(200, 200), cmap=plt.cm.get_cmap('coolwarm'))
    axes[1].set_title("Proba in")
    im_out = axes[2].imshow(probas_out[0].reshape(200, 200), cmap=plt.cm.get_cmap('coolwarm'))
    axes[2].set_title("Proba out")
    im_r = axes[3].imshow(logistic_ratios[0], cmap=plt.cm.get_cmap('coolwarm'))
    axes[3].set_title("log ratio")

    image_selector = Slider(ax=fig.add_axes((.1, .03, .8, .02)), label="Image index", valmin=0,
                            valmax=n_iteration - 1, valinit=0, valfmt="%d", valstep=1)

    contour_selector = Slider(ax=fig.add_axes((.1, .06, .8, .02)), label="contour level", valmin=0,
                              valmax=1, valinit=0.5, valfmt="%3.2f", valstep=.05)

    def update(index):
        update_contour(contour_selector.val)
        index = int(index)
        im_in.set_data(probas_in[index].reshape(200, 200))
        im_out.set_data(probas_out[index].reshape(200, 200))
        im_r.set_data(logistic_ratios[index])
        axes[0].set_title("Raw STEP %d" % index)

    def update_contour(level):
        for t_contour in contour[0].collections:
            t_contour.remove()
        contour[0] = axes[0].contour(logistic_ratios[int(image_selector.val)], [level], alpha=.5, colors='r')

    image_selector.on_changed(update)
    contour_selector.on_changed(update_contour)
    plt.show()
    return image_selector, contour_selector


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize a numpy array, to put it in [0, 1]. """
    return (x - x.min()) / (x.max() - x.min())


if __name__ == '__main__':

    # Choose the experiment you want to run here!
    image_type = "stack"
    
    if image_type == "stack":

        #object = np.array(Image.open("%s/raw_1.tiff" % project_dir).convert(mode='I'), dtype='float32')
        object = io.imread('./cropped_stack.tif')

        image_to_run = object[50:150,300:400,200:300]
        
        #image_to_run = object[100:179,40:360,60:350]
        phi_init = local_max(image_to_run, 10,5)
        #phi_init = dense_initialization(17,8,image_to_run.shape)
        
        #phi_init = dense_initialization(17,8,image_to_run.shape)
    elif image_type == "ltr":

        project_dir = "data"
        #object = np.array(Image.open("%s/raw_1.tiff" % project_dir).convert(mode='I'), dtype='float32')
        
        #object = io.imread('/Users/bridget/new/ltr.tif')
        #image_to_run = object.astype(float)
        image_to_run = np.load('thres3.npy')
        #phi_init = np.load('lcphi.npy')
        phi_init = local_max(image_to_run,10,5)
        
    elif image_type == "compile":
        object = io.imread('./cropped_stack.tif')

        #image_to_run = object[0:25,200:400,200:400]
        image_to_run = object[0:99,40:360,60:350]
        phi_init = np.load("compile1fc.npy")
    

    else:
        raise ValueError

    flow = MichailovichFlow(image_to_run,
                            energy_type="bhatt",
                            narrowband_type="neighborhood",
                            narrowband_neighborhood=7,
                            alpha_curvature=1e-5,
                            narrowband_threshold=.1,
                            dirac_delta=15,
                            n_samples=5,
                            initial_segmentation=phi_init,
                            kernel="epanechnikov")  #precision 

    d = flow.run_flow(15,5, kde_method="sklearn-no-cv", boundary_type="dirac",  # or dirac
                      never_break_when_energy_increases=True, verbose=20, display=True, save_debug=True,
                      signed_distance_reset_period=10, proba_estimate_reset_period=1000,
                      save_file=True, save_dir="save/test1")
   