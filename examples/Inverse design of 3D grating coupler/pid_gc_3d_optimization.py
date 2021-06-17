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

def runGratingOptimization(gc, initial_params, bounds, wavelengths, polarization, working_dir):

    ## Wrapper function to pass the grating coupler class (could be an inline lambda as well
    def gen_grating(params, fdtd, only_update):
        num_rings, r0, theta_taper, grating_w, connector_pts = gc.unpack_grating_parameters_without_ellipticity(params)

        if not only_update:
            gc.add_rings(fdtd, num_rings, theta_taper, group_name="rings")  
            gc.add_connector(fdtd)
            
        gc.update_rings(fdtd, theta_taper, r0, grating_w, group_name="rings")
        gc.update_connector(fdtd, connector_pts, theta_taper)
    
    geometry = ParameterizedGeometry(func=gen_grating, initial_params=initial_params, bounds=bounds, dx=1e-3) #, deps_num_threads=6

    ######## DEFINE FIGURE OF MERIT ########
    mode_number = 'fundamental TE mode' if polarization == 'TE' else 'fundamental TM mode'
    fom = ModeMatch(monitor_name = 'fom', mode_number = mode_number, direction = 'Backward', target_T_fwd = lambda wl: np.ones(wl.size), norm_p = 1)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter = 250, method = 'L-BFGS-B', scaling_factor = 1, ftol=1e-6, pgtol = 1e-6)

    ######## PUT EVERYTHING TOGETHER ########
    opt = Optimization(base_script = gc.setup_gratingcoupler_3d_base_project, wavelengths = wavelengths, fom = fom, geometry = geometry, optimizer = optimizer, hide_fdtd_cad = False, use_deps = True)

    ######## RUN THE OPTIMIZER ########
    return opt.run(working_dir=working_dir)
    



if __name__ == "__main__":

    n_bg=1.44401           #< Refractive index of the background material (cladding)
    n_wg=3.47668           #< Refractive index of the waveguide material (core)
    lambda0=1550e-9     
    bandwidth = 0e-9
    polarization = 'TE' 
    wg_width=500e-9
    wg_height=220e-9
    etch_depth=80e-9
    theta_fib_mat = 5 #< Angle of the fiber mode in material 
    min_feature_size = 0.1  #< Minimal feature size in um. Set to 0.15 to enforce 150nm min features size!
    theta_taper=30
    perform_pos_sweep = True
    perform_angle_sweep = True

    # Position sweep bounds
    x_min = 11e-6
    x_max = 14e-6

    initial_file = "pid_gc_3d_initial.json"
    output_file = "pid_gc_3d_final.json"

    if os.path.exists(os.path.join(cur_path, initial_file)):
        with open(os.path.join(cur_path, initial_file), "r") as fh:
            data = json.load(fh, cls=LumDecoder)
    else:
        sys.exit("Json file doesn't exist: {0}".format(initial_file))
   
    gc = GratingCoupler(lambda0=lambda0,
                        n_trenches = 25,
                        n_bg=n_bg,
                        n_wg=n_wg,
                        wg_height=wg_height,
                        wg_width=wg_width,
                        etch_depth=etch_depth,
                        theta_fib_mat = theta_fib_mat,
                        polarization=polarization,
                        dx = 30e-9,
                        dzFactor=3,
                        dim=3)
    
    num_rings, r0, theta_taper,  distances, connector_pts = gc.unpack_grating_parameters_without_ellipticity(data)
    
    ## Specify wavelength range to optimize for
    lambda_start = gc.lambda0 - gc.bandwidth/2
    lambda_end   = gc.lambda0 + gc.bandwidth/2
    lambda_pts = 1 if gc.bandwidth==0 else int(gc.bandwidth/5e-9)+1 #< One point per 5nm bandwidth
    wavelengths = Wavelengths(start = lambda_start, stop = lambda_end, points = lambda_pts)


    initial_params=gc.pack_grating_parameters_without_ellipticity(r0, theta_taper, distances, connector_pts)
    bounds = [(min_feature_size, 1)]*(len(initial_params))
    bounds[0] = (0.8,1.6)    #< Bounds for the stating position
    bounds[1] = (0.5,5.)     #< Bounds for the taper angle (in units of 10 deg to keep numbers closer 1)
    bounds[2:(2*num_rings+2)] = [(min_feature_size, 1)]*(len(distances))      #< Bounds for the offset shifting
    bounds[(2*num_rings+2):] = [(min_feature_size/2, 5.0)]*(len(connector_pts))

    working_dir = os.path.join(cur_path,'PID_GC_3D')
    best_fom, best_params = runGratingOptimization(gc, initial_params=initial_params, bounds=bounds, wavelengths=wavelengths, polarization=polarization, working_dir=working_dir)

    new_params = { "fom": best_fom,
               "params": best_params }

    with open(os.path.join(cur_path, output_file), "w") as fh:
        json.dump(new_params, fh, indent=4, cls=LumEncoder)
