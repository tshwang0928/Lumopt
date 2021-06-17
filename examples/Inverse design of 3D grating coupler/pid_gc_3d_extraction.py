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

import matplotlib
import matplotlib.pyplot as plt

from pid_gc_3d_base_project_setup import GratingCoupler

import lumapi
from lumjson import LumEncoder, LumDecoder

if __name__ == "__main__":
    n_bg=1.44401           #< Refractive index of the background material (cladding)
    n_wg=3.47668           #< Refractive index of the waveguide material (core)
    lambda0=1550e-9     
    bandwidth = 100e-9
    polarization = 'TE' 
    wg_width=500e-9
    wg_height=220e-9
    etch_depth=80e-9
    theta_fib_mat = 5 #< Angle of the fiber mode in material 

    create_GDS = True
    plot_FOM = True
    plot_backR = True

    input_file = "pid_gc_3d_final.json"
    output_file = "pid_gc_3d_final"

    if os.path.exists(os.path.join(cur_path, input_file)):
        with open(os.path.join(cur_path, input_file), "r") as fh:
            data = json.load(fh, cls=LumDecoder)["params"]
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
                    
    num_rings, r0, theta_taper, grating_w, connector_pts = gc.unpack_grating_parameters_without_ellipticity(data)

    ## Remove the trenches outside of the simulation region
    wall_positions = np.cumsum(np.concatenate(([r0],grating_w)))
    wall_positions_trimmed = wall_positions[wall_positions<gc.x_max]
    if (len(wall_positions_trimmed)%2) == 0: #< If the number is even we need to remove one more to end with a tooth and not a trench!
        wall_positions_trimmed = wall_positions_trimmed[:-1]
    grating_w = np.diff(wall_positions_trimmed)
    wall_positions = np.cumsum(np.concatenate(([r0],grating_w)))
    num_rings=int(len(grating_w)/2)
                                                        
    gc.n_trenches=num_rings
                    
    y_max = wall_positions[-1]*math.sin(math.radians(theta_taper))                    
    gc.y_min = -y_max                   
    gc.y_max =  y_max

    with lumapi.FDTD(hide=False) as fdtd:
        fdtd.newproject()
        gc.setup_gratingcoupler_3d_base_project(fdtd)
        gc.add_rings(fdtd, num_rings, theta_taper, group_name="rings")  
        gc.add_connector(fdtd)           
        gc.update_rings(fdtd, theta_taper, r0, grating_w, group_name="rings")
        gc.update_connector(fdtd, connector_pts, theta_taper)

        if create_GDS:
            gds_filename = output_file + ".gds"
            extract_contour_script = ('idx = getresult("index_xy","index");' +
                                    'level=[(min(idx.index_z)+max(idx.index_z))/2];' +
                                    'idx.addattribute("index_z_real",real(idx.index_z));'+
                                    'contours = getcontour(idx, "index_z_real", level);'+
                                    'polys=contours{1}.polygons;'+
                                    'f = gdsopen("{}", 1e-3, 1e-9);  '.format(gds_filename) + 
                                    'gdsbegincell(f,"Grating Coupler");'+
                                    'for(i = 1:length(polys)){ gdsaddpoly(f, 1, polys{i}); }'+
                                    'gdsendcell(f);'+
                                    'gdsclose(f);')
            fdtd.eval(extract_contour_script)

        if plot_FOM:                                   
            fig1, ax1 = plt.subplots()
            plt.ion()
            plt.show()

            fdtd.setglobalmonitor("frequency points",201)
            fdtd.setglobalsource("wavelength start", lambda0 - bandwidth/2)
            fdtd.setglobalsource("wavelength stop", lambda0 + bandwidth/2)
            fdtd.setnamed("source","enabled",True)
            fdtd.setnamed("source","override global source settings",False)
            fdtd.setnamed("opt_fields","enabled",False)
            fdtd.setnamed("index_xy","enabled",True)

            if fdtd.getnamednumber('fom_mode_exp') < 1:
                fdtd.addmodeexpansion(name='fom_mode_exp',
                                    auto_update=True,
                                    override_global_monitor_settings=False)
                fdtd.setexpansion('fom_mode_exp', 'fom')
                fdtd.set(fdtd.getnamed('fom',['monitor type','x', 'y','y span','z', 'z span']))          
                mode_name = 'fundamental TE mode' if polarization == 'TE' else 'fundamental TM mode'
                fdtd.setnamed('fom_mode_exp', 'mode selection', mode_name)
                fdtd.updatemodes()
                
            if fdtd.getnamednumber('bksource') > 0:
                fdtd.setnamed("bksource","enabled", False)  
            fdtd.save(output_file)
            fdtd.run()
            monitor_result = fdtd.getresult('fom_mode_exp','expansion for fom_mode_exp')
            l=monitor_result['lambda'].flatten()
            T=-monitor_result['T_backward'].flatten()
            TdB=10*np.log10(T)
                        
            TdB_normalized = (TdB-max(TdB))+1  #< To find 1dB bandwidth, shift by peak + 1dB and find zero-crossings
            zero_crossings = np.where(np.diff(np.sign(TdB_normalized)))[0]
            bw1dB=0
            if len(zero_crossings)==2:
                bw1dB = l[zero_crossings[1]]-l[zero_crossings[0]]
                print(" * 1dB-bandwidth: {:4.2f}nm".format(bw1dB*1e9))
                        
                        
            ax1.plot(l*1e6, TdB)           
            ax1.set(xlabel='Wavelength (um)', ylabel='T', title='lambda_0={}, theta={}'.format(lambda0*1e9, theta_fib_mat))
            ax1.autoscale(enable=True, axis='x', tight=True)
            ax1.set_ylim(-10, 0)
            ax1.axvline(x=(lambda0*1e9-bandwidth*1e9/2.)/1000.,color='k')
            ax1.axvline(x=(lambda0*1e9+bandwidth*1e9/2.)/1000.,color='k')
            ax1.grid()
            fig1.savefig(os.path.join(cur_path,output_file+"spectrum_e{:04d}_l{:04d}_b{:04d}_t{:04d}.png".format(int(etch_depth*1e9),
                                                                                        int(lambda0*1e9),
                                                                                        int(bandwidth*1e9),
                                                                                        int(theta_fib_mat*100))))
            peak_val = np.amax(T)
            peak_val_dB= 10*np.log10(peak_val)
            peak_idx = np.where(T == peak_val)[0]
            l_peak = l[peak_idx][0]
            print(" * Peak T: {:1.4f} ({:2.2f}dB) at lambda={:4.2f}nm".format(peak_val,peak_val_dB,l_peak*1e9))

        if plot_backR:
            fig2, ax2 = plt.subplots()
            plt.ion()
            plt.show()
              
            fdtd.switchtolayout()

            fdtd.setglobalmonitor("frequency points",201)
            fdtd.setglobalsource("wavelength start", lambda0 - bandwidth/2)
            fdtd.setglobalsource("wavelength stop", lambda0 + bandwidth/2)
            fdtd.setnamed("opt_fields","enabled",False)
            fdtd.setnamed("index_xy","enabled",True)

            if fdtd.getnamednumber('fom_mode_exp') < 1:
                fdtd.addmodeexpansion(name='fom_mode_exp',
                                    auto_update=True,
                                    override_global_monitor_settings=False)
                fdtd.setexpansion('fom_mode_exp', 'fom')
                fdtd.set(fdtd.getnamed('fom',['monitor type','x', 'y','y span','z', 'z span']))          
                mode_name = 'fundamental TE mode' if polarization == 'TE' else 'fundamental TM mode'
                fdtd.setnamed('fom_mode_exp', 'mode selection', mode_name)
                fdtd.updatemodes()

            mode_type = 'fundamental TM mode' if polarization == 'TE' else 'fundamental TE mode' #< TE and TM is flipped due to cross-section in xy-plane!

            if fdtd.getnamednumber('bksource') == 0:
                fdtd.addmode(name='bksource',mode_selection=mode_type, x=5.5e-6,override_global_source_settings=False, y=0, y_span=3e-6)

            pos_source = (gc.x_min + fdtd.getnamed("fom", "x")) / 2

            fdtd.setnamed("bksource","enabled",True)                       
            fdtd.setnamed("bksource", "x", pos_source)
            fdtd.setnamed("source","enabled",False)

            fdtd.save(output_file+"_backR")
            fdtd.run()

            monitor_result2 = fdtd.getresult('fom_mode_exp','expansion for fom_mode_exp')
            l2=monitor_result2['lambda'].flatten()
            T2=-monitor_result2['T_backward'].flatten()
            T2dB=10*np.log10(T2)
                    
            #plt.draw()
            #plt.pause(0.001)
                
            ax2.plot(l2*1e6, T2dB)
            ax2.set(xlabel='Wavelength (um)', ylabel='T [dB]', title='lambda_0={}, theta={}'.format(lambda0*1e6, theta_fib_mat))
            ax2.autoscale(enable=True, axis='x', tight=True)
            #ax2.set_ylim(-30, 0)
            #ax2.axvline(x=(lambda0-bandwidth/2.)/1000.,color='k')
            #ax2.axvline(x=(lambda0+bandwidth/2.)/1000.,color='k')
            #ax2.axvline(x=(lambda0)/1000.,color='k')
            ax2.grid()
            fig2.savefig(os.path.join(cur_path, output_file+'_spectrum_backR_inDB.png'))
            plt.draw()
            plt.pause(0.001)
                    
            ## Write out some statistics as well:
            peak_val2 = np.amax(T2)
            peak_val2_dB= 10*np.log10(peak_val2)
            print(" * Peak R_back: {:1.4f} ({:2.2f}dB)".format(peak_val2,peak_val2_dB))