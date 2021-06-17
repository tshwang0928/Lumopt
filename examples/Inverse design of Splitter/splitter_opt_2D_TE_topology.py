######## IMPORTS ########
# General purpose imports
import numpy as np
import os
import sys
import scipy as sp

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

######## RUNS TOPOLOGY OPTIMIZATION OF A 2D STRUCTURE ########
def runSim(params, eps_min, eps_max, x_pos, y_pos, filter_R):

    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization2D(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=0, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    # The base simulation script defines a field monitor named 'fom' at the point where we want to modematch to the fundamental TE mode
    fom = ModeMatch(monitor_name = 'fom', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=0.5)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    #optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, target_fom=0.5, scale_initial_gradient_to=0.25)
    optimizer = ScipyOptimizers(max_iter=50, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6, ftol=1e-4, scale_initial_gradient_to=0.25)

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    cur_path = os.path.dirname(os.path.abspath(__file__))
    script = load_from_lsf(os.path.join(cur_path, 'splitter_base_2D_TE_topology.lsf'))
	
    ## Here, we substitute the width of the optimization regio
    size_x = max(x_pos) - min(x_pos)
    script = script.replace('opt_size_x=3.5e-6','opt_size_x={:1.6g}'.format(size_x))
    
    wavelengths = Wavelengths(start = 1450e-9, stop = 1650e-9, points = 11)
    opt = Optimization(base_script=script, wavelengths = wavelengths, fom=fom, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False)

    ######## RUN THE OPTIMIZER ########
    opt.run()

if __name__ == '__main__':
    size_x = 3000		#< Length of the device (in nm). Longer devices typically lead to better performance
    delta_x = 20		#< Size of a pixel along x-axis (in nm)
	
    size_y = 1800		#< Since we use symmetry, this is only have the extent along the y-axis (in nm)
    delta_y = 20		#< Size of a pixel along y-axis (in nm)
	
    filter_R = 500		#< Radius of the smoothing filter which removes small features and sharp corners (in nm)
    
    eps_max = 2.8**2	#< Effective permittivity for a Silicon waveguide with a thickness of 220nm
    eps_min = 1.44**2	#< Permittivity of the SiO2 cladding

    x_points=int(size_x/delta_x)+1
    y_points=int(size_y/delta_y)+1

    x_pos = np.linspace(-size_x/2,size_x/2,x_points)*1e-9
    y_pos = np.linspace(0,size_y,y_points)*1e-9

    ## Set initial conditions
    initial_cond = None                                #< Use the structure 'initial_guess' as defined in the project file

    ## Alternative initial conditions
    #initial_cond = np.ones((x_points,y_points))       #< Start with the domain filled with eps_max
    #initial_cond = 0.5*np.ones((x_points,y_points))   #< Start with the domain filled with (eps_max+eps_min)/2
    #initial_cond = np.zeros((x_points,y_points))      #< Start with the domain filled with eps_min         
    
    runSim(initial_cond, eps_min, eps_max, x_pos, y_pos, filter_R*1e-9)
