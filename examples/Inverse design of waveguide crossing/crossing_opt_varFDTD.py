#############################################################################
# Scriptfile: crossing_opt_varFDTDD.py
#
# Description:
# This script runs the adjoint shape-based optimization for inverse
# design of the SOI waveguide crossing in 2D
#
# Steps include:
# 1. Define the base simulation with lsf file
# 2. Optimizable geometry and optimization parameters
# 3. Run optimization
# 4. Save results
#
# Copyright 2019, Lumerical Solutions, Inc.
# Copyright chriskeraly
##############################################################################

import os,sys
import numpy as np
import scipy as sp
sys.path.append("C:\\Program Files\\Lumerical\\v211\\api\\python\\")
import lumapi

from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.geometries.polygon import FunctionDefinedPolygon
from lumopt.utilities.materials import Material
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

######## BASE SIMULATION ########
crossing_base = load_from_lsf('C:\\Users\\HWANGTAESEUNG\\Documents\\Inverse design of waveguide crossing\\varFDTD_crossing.lsf')

######## DIRECTORY FOR GDS EXPORT #########
example_directory = os.getcwd()

######## SPECTRAL RANGE #########
wavelengths = Wavelengths(start = 1300e-9, stop = 1800e-9, points = 25)

######## OPTIMIZABLE GEOMETRY ########
# The class FunctionDefinedPolygon needs a parameterized Polygon (with points ordered
# in a counter-clockwise direction). Here the geometry is defined by 10 parameters defining
# the knots of a spline, and the resulting Polygon has 100 edges, making it quite smooth.

def cross(params):
    y_end = params[-1]
    x_end = 0 - y_end
    points_x = np.concatenate(([-2.01e-6], np.linspace(-2e-6, x_end, 10)))
    points_y = np.concatenate(([0.25e-6], params))
    n_interpolation_points = 50
    polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)
    interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
    polygon_points_y = [max(min(point, 1e-6), -1e-6) for point in interpolator(polygon_points_x)]
    pplu = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppld = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdl = [(-y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppdr = [(y, x) for x, y in zip(polygon_points_x, polygon_points_y)]
    pprd = [(-x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppru = [(-x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppur = [(y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    ppul = [(-y, -x) for x, y in zip(polygon_points_x, polygon_points_y)]
    polygon_points = np.array(pplu[::-1] + ppld[:-1] + ppdl[::-1] + ppdr[:-1] + pprd[::-1] + ppru[:-1] + ppur[::-1] + ppul[:-1]) #put all the points together in one array
    return polygon_points

initial_params = np.linspace(0.25e-6, 0.6e-6, 10)
bounds = [(0.2e-6, 1e-6)]*initial_params.size
eps_in = Material(name = 'Si: non-dispersive', mesh_order = 2)
eps_out = Material(name = 'SiO2: non-dispersive', mesh_order = 3)
depth = 220.0e-9
polygon = FunctionDefinedPolygon(func = cross,
                                 initial_params = initial_params,
                                 bounds = bounds,
                                 z = 0.0,
                                 depth = depth,
                                 eps_out = eps_out,
                                 eps_in = eps_in,
                                 edge_precision = 5,
                                 dx = 1.0e-9)


######## FIGURE OF MERIT ########
mode_fom = ModeMatch(monitor_name = 'fom',
                     mode_number = 'fundamental mode',
                     direction = 'Forward',
                     multi_freq_src = False,
                     target_T_fwd = lambda wl: np.ones(wl.size),
                     norm_p = 2)

######## OPTIMIZATION ALGORITHM ########
scaling_factor = 1.0e6
scipy_optimizer = ScipyOptimizers(max_iter = 35,
                                  method = 'L-BFGS-B',
                                  scaling_factor = scaling_factor,
                                  pgtol = 1.0e-5,
                                  ftol = 1.0e-5,
                                  scale_initial_gradient_to = 0.0)

######## PUT EVERYTHING TOGETHER ########
opt = Optimization(base_script = crossing_base,
                   wavelengths = wavelengths,
                   fom = mode_fom, geometry = polygon,
                   optimizer = scipy_optimizer,
                   use_var_fdtd = True,
                   hide_fdtd_cad = False,
                   use_deps = True,
                   plot_history = True,
                   store_all_simulations = False)

######## RUN OPTIMIZATION ########
results = opt.run()

######## SAVE THE BEST PARAMETERS TO FILE ########
np.savetxt('C:\\Users\\HWANGTAESEUNG\\Documents\\Inverse design of waveguide crossing\\2D_parameters.txt', results[1])

######## EXPORT OPTIMIZED STRUCTURE TO GDS ########
gds_export_script = str("gds_filename = 'crossing_2D.gds';" +
                        "top_cell = 'model';" +
                        "layer_def = [1, {0}, {1}];".format(-depth/2, depth/2) +
                        "n_circle = 64;" +
                        "n_ring = 64;" +
                        "n_custom = 64;" +
                        "n_wg = 64;" +
                        "round_to_nm = 1;" +
                        "grid = 1e-9;" +
                        "max_objects = 10000;" +
                        "Lumerical_GDS_auto_export;")
                        
with lumapi.MODE(hide = False) as mode:
    mode.cd(example_directory)
    mode.eval(crossing_base)
    mode.addpoly(vertices = cross(results[1]))
    mode.set('x', 0.0)
    mode.set('y', 0.0)
    mode.set('z', 0.0)
    mode.set('z span', depth)
    mode.set('material','Si: non-dispersive')
    mode.eval(gds_export_script)