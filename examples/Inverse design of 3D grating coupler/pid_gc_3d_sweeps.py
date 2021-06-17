#######################################################################
# Copyright (c) 2021 Ansys Inc.
#
#######################################################################

######## IMPORTS ########
# General purpose imports
import os
import math
import sys
import json
import numpy as np
import scipy as sp

cur_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(cur_path)

# Optimization specific imports
from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization
from lumopt.utilities.materials import Material
from pid_gc_3d_base_project_setup import GratingCoupler

from lumjson import LumEncoder, LumDecoder
import lumapi

if __name__ == "__main__":

    n_bg=1.44401           #< Refractive index of the background material (cladding)
    n_wg=3.47668           #< Refractive index of the waveguide material (core)
    lambda0=1550e-9     
    bandwidth = 0e-9
    polarization = 'TE' 
    wg_width=500e-9        #< Waveguide width
    wg_height=220e-9       #< Waveguide height
    etch_depth=80e-9       #< etch depth
    theta_fib_mat = 5      #< Angle of the fiber mode in material 
    theta_taper=30

    perform_pos_sweep = True
    perform_angle_sweep = True

    # Position sweep bounds
    x_min = 10e-6
    x_max = 12e-6


    initial_file = "pid_gc_2d_initial.json"
    output_file = "pid_gc_3d_initial.json"

    if os.path.exists(os.path.join(cur_path, initial_file)):
        with open(os.path.join(cur_path, initial_file), "r") as fh:
            data = json.load(fh, cls=LumDecoder)["initial_params"]
    else:
        sys.exit("Json file doesn't exist: {0}".format(initial_file))

    r0 = data[0]*1e-6
    distances = data[1:]*1e-6

    num_rings = int(round(len(distances)/2))

    gc = GratingCoupler(lambda0=lambda0,
                        n_trenches = num_rings,
                        n_bg=n_bg,
                        n_wg=n_wg,
                        wg_height=wg_height,
                        wg_width=wg_width,
                        etch_depth=etch_depth,
                        theta_fib_mat=theta_fib_mat,
                        polarization=polarization,
                        dx=30e-9,
                        dzFactor=3,
                        dim=3)

    ## Specify wavelength range to optimize for
    lambda_start = gc.lambda0 - gc.bandwidth/2
    lambda_end   = gc.lambda0 + gc.bandwidth/2
    lambda_pts = 1 if gc.bandwidth==0 else int(gc.bandwidth/5e-9)+1 #< One point per 5nm bandwidth
    wavelengths = Wavelengths(start = lambda_start, stop = lambda_end, points = lambda_pts)

    ## If we have not done so already, we should probably sweep the fiber position (and possibly the fiber angle?)    
    if perform_pos_sweep:
        fdtd = lumapi.FDTD(hide = False)
        cur_best_T, r0 = gc.perform_3d_position_sweep(fdtd, num_rings, theta_taper, distances, x_min, x_max, 21, working_dir="sweep_r0") #, basefilename=basefilename)
        print("New best position is x={} with T={}".format(r0,cur_best_T))
        fdtd.close()
        
    if perform_angle_sweep:
        fdtd = lumapi.FDTD(hide = False)
        cur_best_T, theta_taper = gc.perform_taper_angle_sweep(fdtd, num_rings, r0, distances, theta_taper-3, theta_taper+2, 11, working_dir="sweep_theta")
        print("New best taper angle is theta={} with T={}".format(theta_taper,cur_best_T))
        fdtd.close()

    initial_points_y = np.linspace(gc.wg_width/2.0, gc.initial_points_x[-1]*math.tan(math.radians(theta_taper)), gc.n_connector_pts+2)
    connector_pts = initial_points_y[1:-1] #< Use units of um to bring to same order of magnitude as other paramters! First and last point remain fixed!

    new_params = gc.pack_grating_parameters_without_ellipticity(r0, theta_taper, distances, connector_pts)

    with open(os.path.join(cur_path, output_file), "w") as fh:
        json.dump(new_params, fh, indent=4, cls=LumEncoder)