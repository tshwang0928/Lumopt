#######################################################################
# Copyright (c) 2021 Ansys Inc.
#
#######################################################################

######## IMPORTS ########
# General purpose imports
import os
import lumapi
import math
import numpy as np
import scipy as sp
import scipy.interpolate

import json
from lumjson import LumEncoder, LumDecoder
from collections import OrderedDict

class GratingCoupler:
    """Holds basic parameters of the grating coupler to optimize"""
    
    def __init__(self, lambda0, n_trenches, n_bg=1.44401, mat_bg="<Object defined dielectric>", n_wg=3.47668, mat_wg="<Object defined dielectric>", bandwidth=0, wg_height=220e-9, etch_depth=70e-9, wg_width=450e-9, theta_fib_mat = 8, dx = 30e-9, dzFactor=3, dim=2, polarization = 'TE', initial_theta_taper = 30, optim=False):
        self.lambda0 = lambda0
        self.bandwidth = bandwidth
        self.n_trenches = n_trenches
        
        self.wg_height  = wg_height
        self.etch_depth = etch_depth
        self.wg_width   = wg_width    #< Only matters for 3D simulation
        self.material_name = mat_wg
        self.index_wg = n_wg
        self.n_bg = n_bg # background refractive index
        self.mat_bg = mat_bg
        self.x_fib = 18e-6
        self.x_fib_span = 26e-6 #< Roughly 2.5 * mode diameter
        self.z_fib = 0.5e-6
        
        self.mode_pos_x = self.x_fib-self.x_fib_span/2 - 1e-6 if dim==2 else -1e-6
        self.mode_span_y = 3e-6
        self.mode_span_z = 3e-6
        
        self.bksrc_pos_x = self.mode_pos_x+100e-9

        self.dzFactor = dzFactor
        self.dx = dx
        self.dy = dx
        self.dz = etch_depth/dzFactor
        
        ## Dimension of the simulation region
        self.x_min = self.mode_pos_x - 5*self.dx
        self.x_max = self.x_fib+self.x_fib_span/2 + 1e-6
        self.y_min =-self.x_fib_span/2
        self.y_max = self.x_fib_span/2
        self.z_min = -2.05e-6
        self.z_max = 1.5e-6
        
        self.x_min_opt_region = self.x_fib-self.x_fib_span/2. if dim==2 else self.mode_pos_x+5*dx

        #theta_fib_air = 10
        #theta_fib_mat = math.degrees(math.asin(math.sin(math.radians(theta_fib_air))/n_bg))
        self.theta_fib_mat = theta_fib_mat #math.degrees(math.asin(math.sin(math.radians(theta_fib_air))/n_bg))
        self.theta_fib_air = math.degrees(math.asin(math.sin(math.radians(self.theta_fib_mat))*self.n_bg))
        
        self.F0 = 0.95           #< Starting value for the filling factor. Could be up to 1 but that would lead to very narrow trenches which can't be manufactured. 

        self.x_connector_start =-0.5e-6
        self.x_connector_end   = 4.0e-6
        self.n_connector_pts = 28
        self.initial_points_x = np.linspace(self.x_connector_start, self.x_connector_end, self.n_connector_pts+2)    #< x-range for the connector region

        self.pol_angle = 90 if polarization =='TE' else 0
        self.initial_theta_taper = initial_theta_taper

        self.optim = optim

    
    def setup_gratingcoupler_3d_base_project(self, fdtd):
        """
        Setup the basic 3D FDTD project with the simulation region, source, monitors, etc.
        """  
        
        ## CLEAR SESSION
        #fdtd.clear()
        fdtd.newproject()
        
        ## Start adding base components
        fdtd.redrawoff()

        ## Set FDTD properties
        props = OrderedDict([
            ("dimension", "3D"),
            ("x min", self.x_min),
            ("x max", self.x_max), 
            ("y min", self.y_min), 
            ("y max", self.y_max), 
            ("z min", self.z_min), 
            ("z max", self.z_max), 
            ("background material", self.mat_bg),
            ("y min bc", "anti-symmetric"),
            ("simulation time", 5000e-15), 
            ("auto shutoff min", 1e-6),
            ("mesh refinement", "conformal variant 0"),
            ("meshing tolerance", 1.2e-15), 
            ("use legacy conformal interface detection", False)
            ])

        if self.mat_bg == "<Object defined dielectric>":
            props["index"] = self.n_bg

        if self.optim:
            props["mesh refinement"] = "precise volume average"
            props["meshing refinement"] = 11

        if self.pol_angle == 0:
            props["y min bc"] = "symmetric"

        fdtd.addfdtd(properties=props)
                
        fdtd.addgaussian(name="source", injection_axis="z-axis", direction="backward", polarization_angle=self.pol_angle, x=self.x_fib, x_span=self.x_fib_span, y_min=self.y_min, y_max=self.y_max, z=self.z_fib,
                        beam_parameters="Waist size and position", waist_radius_w0=5.2e-6, distance_from_waist=0.0, angle_theta=self.theta_fib_mat,
                        center_wavelength=self.lambda0, wavelength_span=0.1e-6, optimize_for_short_pulse=False)
                        
        fdtd.setglobalsource("center wavelength",self.lambda0)
        fdtd.setglobalsource("wavelength span",0.1e-6)
        fdtd.setglobalsource("optimize for short pulse",False)
        fdtd.setglobalmonitor("frequency points",11)
        fdtd.setglobalmonitor("use wavelength spacing",True)
    
        fdtd.addmesh(name="source_mesh",x=self.x_fib, x_span=24e-6, y_min=self.y_min, y_max=self.y_max, z=self.z_fib, z_span=2*self.dz, override_x_mesh=False, override_y_mesh=False, override_z_mesh=True, dz=self.dz)
        fdtd.setnamed("source_mesh","enabled",False) #< Disable by default but need to check the effect
        
        if self.material_name == "<Object defined dielectric>":
            fdtd.addrect(name="substrate", x_min=(self.x_min-2e-6), x_max=(self.x_max+2e-6), y_min=(self.y_min-2e-6), y_max=(self.y_max+2e-6), z_min=-4e-6, z_max=-2e-6, material=self.material_name, index=self.index_wg, alpha=0.1)
        else:
            fdtd.addrect(name="substrate", x_min=(self.x_min-2e-6), x_max=(self.x_max+2e-6), y_min=(self.y_min-2e-6), y_max=(self.y_max+2e-6), z_min=-4e-6, z_max=-2e-6, material=self.material_name, alpha=0.1)

        
        fdtd.addpower(name="fom", monitor_type="2D X-normal", x=self.mode_pos_x, y=0, y_span=self.mode_span_y, z=0, z_span=self.mode_span_z)
        fdtd.addmesh(name="fom_mesh", x=self.mode_pos_x, x_span=2*self.dx, y=0, y_span=self.mode_span_y, z=0, z_span=self.mode_span_z, override_x_mesh=True, dx=self.dx, override_y_mesh=False, override_z_mesh=False )
    
        fdtd.addpower(name="opt_fields",monitor_type="3D", x_min=self.x_min_opt_region, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z_min=self.wg_height-self.etch_depth, z_max=self.wg_height,
                    output_Hx=False, output_Hy=False, output_Hz=False, output_power=False)
        fdtd.addmesh(name="opt_fields_mesh",               x_min=self.x_min_opt_region, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z_min=self.wg_height-self.etch_depth, z_max=self.wg_height, dx=self.dx, dy=self.dy, dz=self.dz)
        
        fdtd.addindex(name="index_xy", monitor_type="2D Z-normal", x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, z=self.wg_height-(self.etch_depth/2.), spatial_interpolation='none', enabled=False)
        fdtd.addindex(name="index_xz", monitor_type="2D Y-normal", x_min=self.x_min, x_max=self.x_max, y=0, z_min=self.z_min, z_max=self.z_max, spatial_interpolation='none', enabled=False)
        
        
        if self.material_name == "<Object defined dielectric>":
            fdtd.addrect(name='wg', x_min=(self.x_min-2e-6), x_max=2e-6, y=0, y_span=self.wg_width, z_min=0, z_max=self.wg_height, material=self.material_name, index=self.index_wg)
        else:
            fdtd.addrect(name='wg', x_min=(self.x_min-2e-6), x_max=2e-6, y=0, y_span=self.wg_width, z_min=0, z_max=self.wg_height, material=self.material_name)


        theta_start = self.initial_theta_taper
        theta_stop = 360.0 - theta_start

        if self.material_name == "<Object defined dielectric>":
            fdtd.addring(name='silicon', x=0, y=0, z_min=0, z_max=self.wg_height, inner_radius=0, outer_radius=60e-6, theta_start=theta_stop, theta_stop=theta_start, material=self.material_name, index=self.index_wg)
        else:
            fdtd.addring(name='silicon', x=0, y=0, z_min=0, z_max=self.wg_height, inner_radius=0, outer_radius=60e-6, theta_start=theta_stop, theta_stop=theta_start, material=self.material_name)

    
        fdtd.redrawon()

    
    def get_opt_region_pos(self):
        ## Working with integers (unit nm) to avoid rounding problems
        x_min_in_nm=self.x_min_opt_region*1e9
        x_max_in_nm=self.x_max*1e9
        y_min_in_nm=0
        y_max_in_nm=self.y_max*1e9
        z_max_in_nm=self.wg_height*1e9
        z_min_in_nm=(self.wg_height-self.etch_depth)*1e9
        
        dx_in_nm = int(round(self.dx*1e9))
        dy_in_nm = int(round(self.dy*1e9))
        print("{} {}".format(dx_in_nm,dy_in_nm))
        x_points = int(round(x_max_in_nm-x_min_in_nm)/dx_in_nm)+1
        y_points = int(round(y_max_in_nm-y_min_in_nm)/dy_in_nm)+2
        z_points = self.dzFactor+2
        print("{} {} {}".format(x_points,y_points,z_points))
        x_pos = np.linspace(x_min_in_nm*1e-9,x_max_in_nm*1e-9,x_points)
        y_pos = np.linspace(y_min_in_nm*1e-9,y_max_in_nm*1e-9,y_points)
        z_pos = np.linspace(z_min_in_nm*1e-9,z_max_in_nm*1e-9,z_points)
        return x_pos, y_pos, z_pos


    def get_permittivities(self, fdtd, wavelengths): 
        eps_min=self.n_bg*self.n_bg
        
        fdtd.setglobalsource("wavelength start",wavelengths.min())
        fdtd.setglobalsource("wavelength stop",wavelengths.max())
        f0 = fdtd.getglobalsource('center frequency')
        n_mat = fdtd.getindex(self.material_name,f0)[0]
        eps_max=n_mat*n_mat
        return eps_min, eps_max[0].real
    

    def pack_grating_parameters(self, r0, theta_taper,  distances, ellipticity, connector_pts):
        return np.concatenate((np.array([r0*1e5, theta_taper/10.]), distances*1e-6, ellipticity, connector_pts*1e-6))


    def unpack_grating_parameters(self, params):
        num_p = int( (len(params)-3-self.n_connector_pts)/2)
        
        r0 = params[0]*1e-5                 #< Units are 10um, so just scale by 1e-5 instead of 1e-6!
        theta_taper = params[1]*10
        x_offset = params[2]               # Offset of the focus point to the start of the waveguide
        grating_w = params[3:(num_p+3)]*1e-6
        ellipticity = params[(num_p+3):(2*num_p+2)]
        connector_pts = params[(2*num_p+3):]*1e-6
        num_rings = int(len(grating_w)/2)
        
        return num_rings, r0, theta_taper, x_offset, grating_w, ellipticity, connector_pts
    
    
    def pack_grating_parameters_without_ellipticity(self, r0, theta_taper,  distances, connector_pts):
        return np.concatenate((np.array([r0*1e5, theta_taper/10.]), distances*1e6, connector_pts*1e6))


    def unpack_grating_parameters_without_ellipticity(self, params):
        num_p = (len(params)-2-(self.n_connector_pts))
        
        r0 = params[0]*1e-5                     #< Units are 10um, so just scale by 1e-5 instead of 1e-6!
        theta_taper = params[1]*10              #< Units are 10 degrees to have number of order 1
        grating_w = params[2:(num_p+2)]*1e-6
        connector_pts = params[(num_p+2):]*1e-6
        num_rings = int(num_p/2)
        
        return num_rings, r0, theta_taper,  grating_w, connector_pts
        
 
    def add_rings(self, fdtd, num_rings, theta_taper, group_name="rings"):
    
        theta_start = -theta_taper
        theta_stop = theta_taper
        x_offset = 0
        self.n_trenches = num_rings
        fdtd.redrawoff()
        ## Base taper is added in setup but angle is adjusted here
        #fdtd.addring(name='silicon', x=0, y=0, z_min=0, z_max=self.wg_height, inner_radius=0, outer_radius=60e-6, theta_start=theta_stop, theta_stop=theta_start, material=self.material_name)
        fdtd.setnamed('silicon', 'theta start', theta_start)
        fdtd.setnamed('silicon', 'theta stop', theta_stop)
        fdtd.addgroup(name=group_name)
    
        ## Add the circles for the trenches from outside inwards
        for etchIdx in range(num_rings-1, -1, -1):
            if self.mat_bg == "<Object defined dielectric>":
                fdtd.addring(name='ring_'+str(etchIdx), 
                             x=x_offset, 
                             y=0, 
                             z_min=self.wg_height-self.etch_depth, 
                             z_max=self.wg_height,
                             theta_start=theta_start,
                             theta_stop=theta_stop,
                             material=self.mat_bg,
                             index=self.n_bg)
            else:
                fdtd.addring(name='ring_'+str(etchIdx), 
                             x=x_offset, 
                             y=0, 
                             z_min=self.wg_height-self.etch_depth, 
                             z_max=self.wg_height,
                             theta_start=theta_start,
                             theta_stop=theta_stop,
                             material=self.mat_bg)

            fdtd.addtogroup(group_name)
        fdtd.redrawon()
    
    
    def convert_params_to_radii(self, r0, distances):
        x = np.cumsum(np.concatenate(([r0],distances)))        
        r1List = x[0:-2:2]
        r2List = x[1::2]
    
        return r1List, r2List, x[-1]
    
    
    def update_rings(self, fdtd, theta_taper, r0, distances, group_name="rings"):
    
        theta_start = -theta_taper
        theta_stop = theta_taper
        x_offset = 0
        r1List, r2List, rEnd = self.convert_params_to_radii(r0, distances)
        #print(r1List)
        fdtd.redrawoff()
        fdtd.setnamed('silicon', 'theta start', theta_start)
        fdtd.setnamed('silicon', 'theta stop', theta_stop)
        fdtd.setnamed('silicon', 'outer radius', rEnd)

        for etchIdx in range((self.n_trenches)-1,-1,-1):
            fdtd.setnamed(group_name+'::ring_'+str(etchIdx), 'x', x_offset )
            fdtd.setnamed(group_name+'::ring_'+str(etchIdx), 'outer radius', r2List[etchIdx] )
            fdtd.setnamed(group_name+'::ring_'+str(etchIdx), 'inner radius', r1List[etchIdx] ) #ellipticity[etchIdx*2]*
            fdtd.setnamed(group_name+'::ring_'+str(etchIdx), 'theta start', theta_start)
            fdtd.setnamed(group_name+'::ring_'+str(etchIdx), 'theta stop', theta_stop)
    
        fdtd.redrawon()    
        
        
    def get_connector_points(self, connector_params, theta_taper):
        """ Creates the vertices of the polygon which connects the waveguide to the taper.
        """
        points_x = np.concatenate(([self.initial_points_x.min() - 0.05e-6], self.initial_points_x, [self.initial_points_x.max() + 0.05e-6]))
        points_y = np.concatenate(([self.wg_width/2.0, self.wg_width/2.0], connector_params, points_x[-2:]*math.tan(math.radians(theta_taper)) ))
        n_interpolation_points = 300
        polygon_points_x = np.linspace(min(points_x), max(points_x), n_interpolation_points)

        interpolator = sp.interpolate.interp1d(points_x, points_y, kind = 'cubic')
        polygon_points_y = interpolator(polygon_points_x)
        polygon_points_up = [(x, y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points_down = [(x, -y) for x, y in zip(polygon_points_x, polygon_points_y)]
        polygon_points = np.array(polygon_points_up[::-1] + polygon_points_down)
        
        return polygon_points
    
            
    def add_connector(self, fdtd, group_name=None):      
        ## Add a block of oxide to hide the original connection
        if self.mat_bg == "<Object defined dielectric>":
            fdtd.addrect(name='blocker', x_min=self.initial_points_x.min(), x_max=self.initial_points_x.max(), y_min=-5e-6, y_max=5e-6, z_min=0, z_max=self.wg_height, index=self.n_bg)
        else:
            fdtd.addrect(name='blocker', x_min=self.initial_points_x.min(), x_max=self.initial_points_x.max(), y_min=-5e-6, y_max=5e-6, z_min=0, z_max=self.wg_height, material=self.mat_bg)
        if group_name is not None:
            fdtd.addtogroup(group_name)
        
        ## Add the polygon with the smooth, variable connector
        if self.material_name == "<Object defined dielectric>":
            fdtd.addpoly(name='connector', x=0, y=0, z_min=0e-9, z_max=self.wg_height, material=self.material_name, index=self.index_wg)
        else:
            fdtd.addpoly(name='connector', x=0, y=0, z_min=0e-9, z_max=self.wg_height, material=self.material_name)
        if group_name is not None:
            fdtd.addtogroup(group_name)
    
    
    def update_connector(self, fdtd, connector_pts, theta_taper, group_name=None):
        ## A spline-like transition between waveguide and taper as well:
        poly_vertices = self.get_connector_points(connector_pts, theta_taper)
        connector_name = 'connector' if group_name is None else group_name+'::connector'
        fdtd.setnamed(connector_name, 'vertices', poly_vertices)    

    
    def perform_taper_angle_sweep(self, fdtd, num_rings, r0, distances, theta_start, theta_end, num_pts, working_dir=None):
        self.setup_gratingcoupler_3d_base_project(fdtd)

        self.add_rings(fdtd, num_rings, theta_start, group_name="rings")  
        self.add_connector(fdtd)
        
        if working_dir is not None:
            if os.path.isfile(working_dir):
                sys.exit("{0} is a file, not a directory".format(working_dir))
            if not os.path.isdir(working_dir):
                os.mkdir(working_dir)
            fdtd.cd(working_dir)
        basefilename = 'theta_sweep'

        for theta_taper in np.linspace(theta_start, theta_end, num_pts):

            self.update_rings(fdtd, theta_taper, r0, distances, group_name="rings")

            initial_points_y = np.linspace(self.wg_width/2.0, self.initial_points_x[-1]*math.tan(math.radians(theta_taper)), self.n_connector_pts+2)
            connector_pts = initial_points_y[1:-1] #< Use units of um to bring to same order of magnitude as other paramters! First and last point remain fixed!
            self.update_connector(fdtd, connector_pts, theta_taper)
            filename = basefilename+'_t{:04d}'.format(int(theta_taper*100))
            fdtd.setnamed('opt_fields','enabled',False)
            fdtd.save(filename)
            fdtd.addjob(filename)

        fdtd.runjobs()
        
        cur_best_T = 0    
        cur_best_theta = theta_start
        for theta_taper in np.linspace(theta_start, theta_end, num_pts):

            filename = basefilename+'_t{:04d}'.format(int(theta_taper*100))
            fdtd.load(filename)                    
            result = fdtd.getresult('fom','T')
            T=result['T']
            idx = int((len(T)-1)/2)  #< Center frequency point
            T0 = abs(T[idx])
            print('theta={}, T={}'.format(theta_taper,T0))
            
            if T0>cur_best_T:
                cur_best_T = T0
                cur_best_theta = theta_taper

        if working_dir is not None:
            fdtd.cd("..")

        return cur_best_T, cur_best_theta   


    def perform_3d_position_sweep(self, fdtd, num_rings, theta_taper, distances, r0_start, r0_end, num_pts, working_dir=None):
        self.setup_gratingcoupler_3d_base_project(fdtd)

        self.add_rings(fdtd, num_rings, theta_taper, group_name="rings")  
        self.add_connector(fdtd)

        if working_dir is not None:
            if os.path.isfile(working_dir):
                sys.exit("{0} is a file, not a directory".format(working_dir))
            if not os.path.isdir(working_dir):
                os.mkdir(working_dir)
            fdtd.cd(working_dir)
        basefilename = 'r0_sweep'

        for r0 in np.linspace(r0_start, r0_end, num_pts):

            self.update_rings(fdtd, theta_taper, r0, distances, group_name="rings")

            initial_points_y = np.linspace(self.wg_width/2.0, self.initial_points_x[-1]*math.tan(math.radians(theta_taper)), self.n_connector_pts+2)
            connector_pts = initial_points_y[1:-1] #< Use units of um to bring to same order of magnitude as other paramters! First and last point remain fixed!
            self.update_connector(fdtd, connector_pts, theta_taper)
            filename = basefilename+'_r{:04d}'.format(int(r0*1e7))
            fdtd.setnamed('opt_fields','enabled',False)
            fdtd.save(filename)
            fdtd.addjob(filename)

        fdtd.runjobs()
        
        cur_best_T = 0    
        cur_best_r0 = r0_start
        for r0 in np.linspace(r0_start, r0_end, num_pts):

            filename = basefilename+'_r{:04d}'.format(int(r0*1e7))
            fdtd.load(filename)                    
            result = fdtd.getresult('fom','T')
            T=result['T']
            idx = int((len(T)-1)/2)  #< Center frequency point
            T0 = abs(T[idx])
            print('r0={}um, T={}'.format(r0*1e6,T0))
            
            if T0>cur_best_T:
                cur_best_T = T0
                cur_best_r0 = r0

        if working_dir is not None:
            fdtd.cd("..")
            
        return cur_best_T, cur_best_r0   
    
  
  
  
if __name__ == "__main__":
    ## For testing purpose
    n_bg = 1.44              #< Refractive index of the background

    gc = GratingCoupler(lambda0=1310e-9,
                        n_trenches = 30,
                        mat_bg="SiO2 (Glass) - Palik",
                        mat_wg="Si (Silicon) - Palik",
                        wg_height=220e-9,
                        wg_width=500e-9,
                        etch_depth=70e-9,
                        theta_fib_mat = 8,
                        dx = 30e-9,
                        dzFactor=3,
                        polarization="TM",
                        dim=3)

    theta_taper=30
    r0 = 15
    distances=[0.15      , 0.34847045, 0.15006019, 0.36481473, 0.15      , 0.33598235, 0.15      , 0.33050892, 0.15      , 0.32504063,
               0.15      , 0.32870693, 0.15650243, 0.33573834, 0.15413071, 0.3306162 , 0.16418292, 0.32418348, 0.16811459, 0.3114108 ,
               0.17915084, 0.30320305, 0.18523447, 0.29430099, 0.19879006, 0.2888649 , 0.20861187, 0.28114498, 0.21807943, 0.27314918,
               0.22624671, 0.2634418 , 0.23484007, 0.25436145, 0.24328793, 0.24471254, 0.25266662, 0.2356394 , 0.26330398, 0.22637478,
               0.2742431 , 0.21689982, 0.28531121, 0.20777701, 0.29617034, 0.19893738, 0.30633181, 0.18935849, 0.31784782, 0.18026158,
               0.32793216, 0.17002477, 0.33925252, 0.16047791, 0.34964887, 0.15080538, 0.3607801 , 0.15      , 0.37217914, 0.15      ,
               0.38311564, 0.15      , 0.39408063, 0.15000575]
     
    r0 = r0*1e-6
    distances = np.array(distances)*1e-6

    test_file = "pid_optim_final_min_feature.json"

    with open(test_file, "r") as fh:
        data = json.load(fh, cls=LumDecoder)["initial_params"]

    print(data)

    r0 = (data[0]+17)*1e-6
    distances = data[1:]*1e-6

    num_rings = int(round(len(distances)/2))
    initial_points_y = np.linspace(gc.wg_width/2.0, gc.initial_points_x[-1]*math.tan(math.radians(theta_taper)), gc.n_connector_pts+2)
    connector_pts = initial_points_y[1:-1] #< Use units of um to bring to same order of magnitude as other paramters! First and last point remain fixed!


    with lumapi.FDTD(hide=False) as fdtd:        
    #fdtd = lumapi.FDTD(hide = False)

        gc.setup_gratingcoupler_3d_base_project(fdtd)

        gc.add_rings(fdtd, num_rings, theta_taper, group_name="rings")  
        gc.update_rings(fdtd, theta_taper, r0, distances, group_name="rings")

        gc.add_connector(fdtd)
        gc.update_connector(fdtd, connector_pts, theta_taper)
    
        input("just waiting...")

        
        #gc.perform_taper_angle_sweep(fdtd, num_rings, r0, distances, theta_start=18, theta_end=24.5, num_pts=14)
        #gc.perform_taper_angle_sweep(fdtd, num_rings, r0, distances, theta_start=25, theta_end=26, num_pts=3)
        #gc.perform_3d_position_sweep(fdtd, num_rings, theta_taper, distances, 13.6, 14.0, 3)