#############################################################################
# Python module: FDTD_y_branch.py 
#
# Description:
# This module defines the y_brach_init_() function used in the 
# optimization for inverse design of the SOI Y-branch in 3D
#
# Steps include:
# 1. Define the base simulation parameters 
# 2. Define the geometry of input and output waveguides
# 3.Set up source and monitors and simulation region
# 
# Copyright 2019, Lumerical Solutions, Inc.
##############################################################################

######## IMPORTS ########
# General purpose imports
import lumapi
import numpy as np
from scipy.constants import c

def y_branch_init_(fdtd): 
           
    
	## CLEAR SESSION
    fdtd.switchtolayout()
    fdtd.selectall()
    fdtd.delete()
    
    ## SIM PARAMS
    size_x=3e-6;
    size_y=3e-6;
    size_z=1.2e-6;
    mesh_x=20e-9;
    mesh_y=20e-9;
    mesh_z=20e-9;
    finer_mesh_size=2.5e-6;
    finer_mesh_size_z=0.6e-6;
    mesh_accuracy=4;
    lam_c = 1.550e-6;
    
    # MATERIAL
    opt_material=fdtd.addmaterial('Dielectric');
    fdtd.setmaterial(opt_material,'name','Si: non-dispersive');
    n_opt = fdtd.getindex('Si (Silicon) - Palik',c/lam_c);
    fdtd.setmaterial('Si: non-dispersive','Refractive Index',n_opt);
    
    sub_material=fdtd.addmaterial('Dielectric');
    fdtd.setmaterial(sub_material,'name','SiO2: non-dispersive');
    n_sub = fdtd.getindex('SiO2 (Glass) - Palik',c/lam_c);
    fdtd.setmaterial('SiO2: non-dispersive','Refractive Index',n_sub);
    fdtd.setmaterial('SiO2: non-dispersive',"color", np.array([0, 0, 0, 0]));
    
    ## GEOMETRY
    
    #INPUT WAVEGUIDE
    
    fdtd.addrect();
    fdtd.set('name','input wg');
    fdtd.set('x span',3e-6);
    fdtd.set('y span',0.5e-6);
    fdtd.set('z span',220e-9);
    fdtd.set('y',0);
    fdtd.set('x',-2.5e-6);
    fdtd.set('z',0);
    fdtd.set('material','Si: non-dispersive');
    
    #OUTPUT WAVEGUIDES
    
    fdtd.addrect();
    fdtd.set('name','output wg top');
    fdtd.set('x span',3e-6);
    fdtd.set('y span',0.5e-6);
    fdtd.set('z span',220e-9);
    fdtd.set('y',0.35e-6);
    fdtd.set('x',2.5e-6);
    fdtd.set('z',0);
    fdtd.set('material','Si: non-dispersive');
    
    fdtd.addrect();
    fdtd.set('name','output wg bottom');
    fdtd.set('x span',3e-6);
    fdtd.set('y span',0.5e-6);
    fdtd.set('z span',220e-9);
    fdtd.set('y',-0.35e-6);
    fdtd.set('x',2.5e-6);
    fdtd.set('z',0);
    fdtd.set('material','Si: non-dispersive');
    
    fdtd.addrect();
    fdtd.set('name','sub');
    fdtd.set('x span',8e-6);
    fdtd.set('y span',8e-6);
    fdtd.set('z span',10e-6);
    fdtd.set('y',0);
    fdtd.set('x',0);
    fdtd.set('z',0);
    fdtd.set('material','SiO2: non-dispersive');
    fdtd.set('override mesh order from material database',1);
    fdtd.set('mesh order',3);
    fdtd.set('alpha',0.8);
    
    ## FDTD
    fdtd.addfdtd();
    fdtd.set('mesh accuracy',mesh_accuracy);
    fdtd.set('dimension','3D');
    fdtd.set('x min',-size_x/2);
    fdtd.set('x max',size_x/2);
    fdtd.set('y min',-size_y/2);
    fdtd.set('y max',size_y/2);
    fdtd.set('z min',-size_z/2.0);
    fdtd.set('z max',size_z/2.0);
    fdtd.set('force symmetric y mesh',1);
    fdtd.set('force symmetric z mesh',1);
    fdtd.set('z min bc','Symmetric');
    fdtd.set('y min bc','Anti-Symmetric');
    
    
    ## SOURCE
    fdtd.addmode();
    fdtd.set('direction','Forward');
    fdtd.set('injection axis','x-axis');
    #fdtd.set('polarization angle',0);
    fdtd.set('y',0);
    fdtd.set("y span",size_y);
    fdtd.set('x',-1.23e-6);
    fdtd.set('center wavelength',lam_c);
    fdtd.set('wavelength span',0);
    fdtd.set('mode selection','fundamental TE mode');
    
    
    ## MESH IN OPTIMIZABLE REGION
    fdtd.addmesh();
    fdtd.set('x',0);
    fdtd.set('x span',finer_mesh_size);
    fdtd.set('y',0);
    fdtd.set('y span',finer_mesh_size);
    fdtd.set('z',0);
    fdtd.set('z span',finer_mesh_size);
    fdtd.set('dx',mesh_x);
    fdtd.set('dy',mesh_y);
    fdtd.set('dz',mesh_z);
    
    ## OPTIMIZATION FIELDS MONITOR IN OPTIMIZABLE REGION
    
    fdtd.addpower();
    fdtd.set('name','opt_fields');
    fdtd.set('monitor type','3D');
    fdtd.set('x',0);
    fdtd.set('x span',(5/6)*size_x);
    fdtd.set('y',0);
    fdtd.set('y span',(5/6)* size_y);
    fdtd.set('z',0);
    fdtd.set('z span',0.4e-6);
    
    ## FOM FIELDS
    
    fdtd.addpower();
    fdtd.set('name','fom');
    fdtd.set('monitor type','2D X-Normal');
    fdtd.set('x',1.23e-6);
    fdtd.set('y',0);
    fdtd.set('y span',size_y);
    fdtd.set('z',0);
    fdtd.set('z span',size_z)
    
    
    
    
    
    
    
    