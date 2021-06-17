######## IMPORTS ########
# General purpose imports
import os
import sys
import numpy as np

# Optimization specific imports
from lumopt import CONFIG
from lumopt.geometries.topology import TopologyOptimization2D
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths

######## DEFINE BASE SIMULATION ########

def runSim(params, eps_bg, eps_wg, x_pos, y_pos, size_x, size_y, filter_R, working_dir, beta):

    ######## DEFINE GEOMETRY ########
    geometry = TopologyOptimization2D(params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=0, filter_R=filter_R, beta=beta)
 
    ######## DEFINE FIGURE OF MERIT FOR EACH OUTPUT WAVEGUIDE ########
    fom1 = ModeMatch(monitor_name = 'fom_1', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=1)
    fom2 = ModeMatch(monitor_name = 'fom_2', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=1)
    fom3 = ModeMatch(monitor_name = 'fom_3', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=1)
    fom4 = ModeMatch(monitor_name = 'fom_4', mode_number = 'Fundamental TE mode', direction = 'Forward', norm_p = 2, target_fom=1)

    ######## DEFINE OPTIMIZATION ALGORITHM ########
    optimizer = ScipyOptimizers(max_iter=400, method='L-BFGS-B', pgtol=1e-6, ftol=1e-5, scale_initial_gradient_to=0.25)

    ######## DEFINE SETUP SCRIPT AND INDIVIDUAL OPTIMIZERS ########
    script = load_from_lsf('C:\\Users\\HWANGTAESEUNG\\Documents\\Inverse Design of 4-channel wavelength demultiplexer using Topology Optimization\\CWDM_splitter_1310_4ch_2D_TE_topology.lsf')
    script = script.replace('opt_size_x=6e-6','opt_size_x={:1.6g}'.format(size_x))
    script = script.replace('opt_size_y=6e-6','opt_size_y={:1.6g}'.format(size_y))

    wavelengths1 = Wavelengths(start = 1265e-9, stop = 1275e-9, points = 11)
    opt1 = Optimization(base_script=script, wavelengths = wavelengths1, fom=fom1, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=True)
    wavelengths2 = Wavelengths(start = 1285e-9, stop = 1295e-9, points = 11)
    opt2 = Optimization(base_script=script, wavelengths = wavelengths2, fom=fom2, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=False)
    wavelengths3 = Wavelengths(start = 1305e-9, stop = 1315e-9, points = 11)
    opt3 = Optimization(base_script=script, wavelengths = wavelengths3, fom=fom3, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=False)
    wavelengths4 = Wavelengths(start = 1325e-9, stop = 1335e-9, points = 11)
    opt4 = Optimization(base_script=script, wavelengths = wavelengths4, fom=fom4, geometry=geometry, optimizer=optimizer, use_deps=False, hide_fdtd_cad=True, plot_history=False, store_all_simulations=False, save_global_index=False)

    ######## PUT EVERYTHING TOGETHER AND RUN ########
    opt = opt1+opt2+opt3+opt4
    opt.run(working_dir = working_dir)

if __name__ == '__main__':
    size_x = 6000
    size_y = 6000
    filter_R = 200e-9

    x_points=int(size_x/20)+1
    y_points=int(size_y/20)+1
    eps_wg = 2.8**2
    eps_bg = 1.44**2
    x_pos = np.linspace(-size_x/2*1e-9,size_x/2*1e-9,x_points)
    y_pos = np.linspace(-size_y/2*1e-9,size_y/2*1e-9,y_points)

    ## We need to pick an initial condition. Many different options:
    params = 0.5*np.ones((x_points,y_points))     #< Start with the domain filled with (eps_wg+eps_bg)/2
    #params = np.ones((x_points,y_points))        #< Start with the domain filled with eps_wg
    #params = np.zeros((x_points,y_points))       #< Start with the domain filled with eps_bg    
    #params = None                                #< Use the structure defined in the project file as initial condition

    working_dir = 'CWDM_splitter_1310_4ch_2D_TE_x{:04d}_y{:04d}_f{:04d}'.format(size_x,size_y,int(filter_R*1e9))
    runSim(params, eps_bg, eps_wg, x_pos, y_pos, size_x*1e-9, size_y*1e-9, filter_R, working_dir=working_dir, beta=1)