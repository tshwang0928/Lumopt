from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

eps0 = sp.constants.epsilon_0


class TopologyOptimization2DParameters(Geometry):

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps=None, min_feature_size = 0):
        self.last_params=params
        self.eps_min=eps_min
        self.eps_max=eps_max
        self.eps = eps
        self.x=x
        self.y=y
        self.z=z
        self.bounds=[(0.,1.)]*(len(x)*len(y))
        self.filter_R = filter_R
        self.eta = eta

        if (min_feature_size<0) or (min_feature_size>(2*filter_R)):
            raise UserWarning('Value of min_feature_size must be between 0 and 2*filter_R ({})'.format(2*filter_R))

        self.eta_d = eta-self.get_delta_eta_from_length(min_feature_size, self.filter_R)
        self.eta_e = eta+self.get_delta_eta_from_length(min_feature_size, self.filter_R)

        self.g_s_hist = []  #< History of the min feature size penalty term for the "solid"
        self.g_v_hist = []  #< History of the min feature size penalty term for the "void"

        self.beta = beta
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 0
        self.depth = z[-1]-z[0] if (hasattr(z, "__len__") and len(z)>1) else 220e-9
        self.beta_factor = 1.2
        self.discreteness = 0

        self.penalty_scaling_beta_threshold = 99e99 if min_feature_size==0 else 12  #< Very large value (always larger than beta) to disable min feature size constraint
        self.penalty_scaling_factor_max = 1e4
        self.penalty_scaling_factor = min(100*np.square(self.beta - self.penalty_scaling_beta_threshold), self.penalty_scaling_factor_max) if self.beta > self.penalty_scaling_beta_threshold else 0

        ## Prep-work for unfolding symmetry to properly enforce min feature sizes at symmetry boundaries. Off by default for now!
        self.symmetry_x = False
        self.symmetry_y = False
        self.unfold_symmetry = self.symmetry_x or self.symmetry_y #< We do want monitors to unfold symmetry but we need to detect if the size does not match the parameters anymore

    def get_delta_eta_from_length(self, target_length, filter_R):
        scaled_length = target_length/(2*filter_R)
        return np.square(scaled_length) if target_length<filter_R else 0.5-np.square(1-scaled_length)

    def use_interpolation(self):
        return True

    def check_license_requirements(self, sim):
        ## Try to execute one of the topology script commands
        try:
            sim.fdtd.eval(('params = struct;'
                            'params.eps_levels=[1,2];'
                            'params.filter_radius = 3;'
                            'params.beta = 1;'
                            'params.eta = 0.5;'
                            'params.dx = 0.1;'
                            'params.dy = 0.1;'
                            'params.dz = 0.0;'
                            'eps_geo = topoparamstoindex(params,ones(5,5));'))
        except:
            raise UserWarning('Could not execute required topology optimization commands. Either the version of FDTD is outdated or the '
                              'license requirements for topology optimization are not fulfilled. Please contact your support or sales representative.')

        return True

    def calc_penalty_term(self, sim, params):
        if self.penalty_scaling_factor==0:
            return 0

        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.eta_e = {5};'
                       'params.eta_d = {6};'
                       'params.dx = {7};'
                       'params.dy = {8};'
                       'params.dz = 0.0;'
                       'min_feature_indicators = topoparamstominfeaturesizeindicator(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.eta_e,self.eta_d,self.dx,self.dy) )
        min_feature_indicators = sim.fdtd.getv("min_feature_indicators").flatten()

        self.g_s_hist.append(min_feature_indicators[0])   #< For logging/debugging, not needed for the actual calculation
        self.g_v_hist.append(min_feature_indicators[1])   #< For logging/debugging, not needed for the actual calculation
        
        penalty_fom = -self.penalty_scaling_factor*np.sum(min_feature_indicators)
        return penalty_fom

    def calc_penalty_gradient(self, sim, params):
        if self.penalty_scaling_factor==0:
            return np.zeros_like(params)

        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.eta_e = {5};'
                       'params.eta_d = {6};'
                       'params.dx = {7};'
                       'params.dy = {8};'
                       'params.dz = 0.0;'
                       'penalty_grad = topoparamstominfeaturesizegradient(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.eta_e,self.eta_d,self.dx,self.dy) )
        penalty_grad = -self.penalty_scaling_factor*sim.fdtd.getv("penalty_grad")
        return penalty_grad.flatten()

    def calc_discreteness(self):
        ''' Computes a measure of discreteness. Is 1 when the structure is completely discrete and less when it is not. '''
        rho = self.calc_params_from_eps(self.eps).flatten()
        return 1 - np.sum(4*rho*(1-rho)) / len(rho)

    def progress_continuation(self):
        self.discreteness = self.calc_discreteness()
        print("Discreteness: {}".format(self.discreteness))

        # If it is sufficiently discrete (99%), we terminate
        if self.discreteness > 0.99:
            return False

        ## Otherwise, we increase beta and keep going
        self.beta *= self.beta_factor
        print('Beta is {}'.format(self.beta))
        
        if self.beta > self.penalty_scaling_beta_threshold:
            self.penalty_scaling_factor = min(100*np.square(self.beta - self.penalty_scaling_beta_threshold), self.penalty_scaling_factor_max)
            print('The penalty scaling factor is {}'.format(self.penalty_scaling_factor))

        return True

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y, z=self.z, depth=self.depth, beta=self.beta, eps=self.eps)

    def calc_params_from_eps(self,eps):
        return np.minimum(np.maximum((eps - self.eps_min) / (self.eps_max-self.eps_min),0),1.0)

    def set_params_from_eps(self,eps):
        self.last_params = self.calc_params_from_eps(eps)

    def extract_parameters_from_simulation(self, sim):
        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",1);')
        eps = get_eps_from_sim(sim.fdtd, unfold_symmetry=False) #< Don't unfold when getting the inital parameters
        sim.fdtd.selectpartial('initial_guess')
        sim.fdtd.eval('set("enabled",0);')

        sim.fdtd.selectpartial('import')
        sim.fdtd.eval('set("enabled",1);')
        reduced_eps = np.real(eps[0])

        self.set_params_from_eps(reduced_eps)


    def get_eps_from_params(self, sim, params):
        rho = np.reshape(params, (len(self.x),len(self.y)))
        self.last_params = rho

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        ## Extend boundary to include effects from fixed structure

        ## Use script function to convert the raw parameters to a permittivity distribution and get the result
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'eps_geo = topoparamstoindex(params,topo_rho);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        eps = sim.fdtd.getv("eps_geo")

        ## Reduce symmetry again (move to cad eventually?)
        if self.symmetry_x:
            shape = rho.shape
            eps = eps[int((shape[0]-1)/2):,:] #np.vstack( (np.flipud(rho)[:-1,:],rho) )
        if self.symmetry_y:
            shape = rho.shape
            eps = eps[:,int((shape[1]-1)/2):] #np.hstack( (np.fliplr(rho)[:,:-1],rho) )

        return eps

    def initialize(self, wavelengths, opt):
        self.opt=opt
        pass

    def update_geometry(self, params, sim):
        self.eps = self.get_eps_from_params(sim, params)
        self.discreteness = self.calc_discreteness()

    def unfold_symmetry_if_applicable(self, rho):
        ## Expand symmetry (either along x- or y-direction)
        if self.symmetry_x:
            rho = np.vstack( (np.flipud(rho)[:-1,:],rho) )
        if self.symmetry_y:
            rho = np.hstack( (np.fliplr(rho)[:,:-1],rho) )
        return rho


    def get_current_params_inshape(self, unfold_symmetry=False):
        return self.last_params

    def get_current_params(self):
        params = self.get_current_params_inshape()
        return np.reshape(params,(-1)) if params is not None else None

    def plot(self,ax_eps):
        ax_eps.clear()
        x = self.x
        y = self.y
        eps = self.eps
        ax_eps.imshow(np.real(np.transpose(eps)), vmin=self.eps_min, vmax=self.eps_max, extent=[min(x)*1e6,max(x)*1e6,min(y)*1e6,max(y)*1e6], origin='lower')

        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')
        return True

    def write_status(self, f):
        f.write(', {:.4f}, {:.4f}'.format(self.beta, self.discreteness))
        
        if len(self.g_s_hist) > 0:
            f.write(', {:.4g}'.format(self.g_s_hist[-1]))
        else:
            f.write(', {:.4g}'.format(0.0))
        
        if len(self.g_v_hist) > 0:
            f.write(', {:.4g}'.format(self.g_v_hist[-1]))
        else:
            f.write(', {:.4g}'.format(0.0))


class TopologyOptimization2D(TopologyOptimization2DParameters):
    '''
    '''
    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z=0, filter_R=200e-9, eta=0.5, beta=1, eps=None, min_feature_size = 0):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps, min_feature_size = min_feature_size)

    @classmethod
    def from_file(cls, filename, z=0, filter_R=200e-9, eta=0.5, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], z = z, filter_R = filter_R, eta=eta, beta=beta, eps=data["eps"])

    def set_params_from_eps(self,eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        super().set_params_from_eps(eps[:,:,0,0,2])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy) +
                      "dF_dEps = pinch(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'dF_dp = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        # topo_grad = sim.fdtd.getv("dF_dp")

        ## Fold symmetry again (on the CAD this time)
        if self.symmetry_x:
            ## To be tested!
            sim.fdtd.eval(('dF_dp = dF_dp({0}:end,:,:);'
                           'dF_dp(2:end,:,:) = 2*dF_dp(:,2:end,:);').format(int((shape[0]+1)/2)))
        if self.symmetry_y:
            shape = rho.shape
            sim.fdtd.eval(('dF_dp = dF_dp(:,{0}:end,:);'
                           'dF_dp(:,2:end,:) = 2*dF_dp(:,2:end,:);').format(int((shape[1]+1)/2)))

        return "dF_dp"


    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        E_forward_dot_E_adjoint = np.atleast_3d(np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1))))

        dF_dEps = 2*self.dx*self.dy*eps0*E_forward_dot_E_adjoint
        
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        fdtd.putv('x_geo',self.x)
        fdtd.putv('y_geo',self.y)
        fdtd.putv('z_geo',np.array([self.z-self.depth/2,self.z+self.depth/2]))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("dx",{});'
                        'set("dy",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),self.dx,self.dy)
            fdtd.eval(mesh_script)

        if eps is not None:
            fdtd.putv('eps_geo',eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'temp=zeros(length(x_geo),length(y_geo),2);'
                    'temp(:,:,1)=eps_geo;'
                    'temp(:,:,2)=eps_geo;'
                    'importnk2(sqrt(temp),x_geo,y_geo,z_geo);')
            fdtd.eval(script)



## Uses a continuous parameter rho in [0,1] instead of the actual epsilon values as parameters. Makes it
## easier to introduce penalization.
class TopologyOptimization3DLayered(TopologyOptimization2DParameters):

    self_update = False

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R = 200e-9, eta = 0.5, beta = 1, min_feature_size = 0):
        super(TopologyOptimization3DLayered,self).__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, eps=None, min_feature_size = min_feature_size)

    @classmethod
    def from_file(cls, filename, filter_R, eta = 0.5, beta = None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], data["z"], filter_R = filter_R, eta=eta, beta=beta)

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y, z=self.z, beta=self.beta, eps=self.eps)

    def set_params_from_eps(self,eps):
        '''
            The raw epsilon of a 3d system needs to be collapsed to 2d first. For now, we just pick the first z-layer
        '''
        midZ_idx=int((eps.shape[2]+1)/2)
        super().set_params_from_eps(eps[:,:,midZ_idx,0,2])

    def calculate_gradients(self, gradient_fields, sim):

        rho = self.get_current_params_inshape()

        ## Perform the dot-product which corresponds to a sum over the last dimension (which is x,y,z-components)
        E_forward_dot_E_adjoint = np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(),axis=-1)))
 
        ## We integrate/sum along the z-direction
        E_forward_dot_E_adjoint_int_z = np.atleast_3d(np.squeeze(np.sum(E_forward_dot_E_adjoint,axis=2)))

        V_cell = self.dx*self.dy*self.dz
        dF_dEps = 2*V_cell*eps0*E_forward_dot_E_adjoint_int_z

        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.putv("dF_dEps", dF_dEps)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'topo_grad = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        topo_grad = sim.fdtd.getv("topo_grad")

        return topo_grad.reshape(-1, topo_grad.shape[-1])


    def calculate_gradients_on_cad(self, sim, forward_fields, adjoint_fields, wl_scaling_factor):
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)

        sim.fdtd.eval("V_cell = {};".format(self.dx*self.dy*self.dz) +
                      "dF_dEps = sum(sum(2.0 * V_cell * eps0 * {0}.E.E * {1}.E.E,5),3);".format(forward_fields, adjoint_fields) +
                      "num_wl_pts = length({0}.E.lambda);".format(forward_fields) +
                      
                      "for(wl_idx = [1:num_wl_pts]){" +
                      "    dF_dEps(:,:,wl_idx) = dF_dEps(:,:,wl_idx) * wl_scaling_factor(wl_idx);" +
                      "}" + 
                      "dF_dEps = real(dF_dEps);")

        rho = self.get_current_params_inshape()
        sim.fdtd.putv("topo_rho", rho)
        sim.fdtd.eval(('params = struct;'
                       'params.eps_levels=[{0},{1}];'
                       'params.filter_radius = {2};'
                       'params.beta = {3};'
                       'params.eta = {4};'
                       'params.dx = {5};'
                       'params.dy = {6};'
                       'params.dz = 0.0;'
                       'dF_dp = topoparamstogradient(params,topo_rho,dF_dEps);').format(self.eps_min,self.eps_max,self.filter_R,self.beta,self.eta,self.dx,self.dy) )
        #topo_grad = sim.fdtd.getv("topo_grad")
        #return topo_grad.reshape(-1, topo_grad.shape[-1])
        return 'dF_dp'

    def add_geo(self, sim, params=None, only_update = False):

        fdtd=sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(sim, params.reshape(-1))

        if not only_update:
            set_spatial_interp(sim.fdtd,'opt_fields','specified position') 
            set_spatial_interp(sim.fdtd,'opt_fields_index','specified position') 

            script=('select("opt_fields");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});'
                    'set("z min",{});'
                    'set("z max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z))
            fdtd.eval(script)

            script=('select("opt_fields_index");'
                    'set("x min",{});'
                    'set("x max",{});'
                    'set("y min",{});'
                    'set("y max",{});'
                    'set("z min",{});'
                    'set("z max",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z))
            fdtd.eval(script)

            script=('addimport;'
                    'set("detail",1);')
            fdtd.eval(script)

            mesh_script=('addmesh;'
                        'set("x min",{});'
                        'set("x max",{});'
                        'set("y min",{});'
                        'set("y max",{});'
                        'set("z min",{});'
                        'set("z max",{});'
                        'set("dx",{});'
                        'set("dy",{});'
                        'set("dz",{});').format(np.amin(self.x),np.amax(self.x),np.amin(self.y),np.amax(self.y),np.amin(self.z),np.amax(self.z),self.dx,self.dy,self.dz)
            fdtd.eval(mesh_script)


        if eps is not None:
            # This is a layer geometry, so we need to expand it to all layers
            full_eps = np.broadcast_to(eps[:, :, None],(len(self.x),len(self.y),len(self.z)))   #< TODO: Move to Lumerical script to reduce transfers

            fdtd.putv('x_geo',self.x)
            fdtd.putv('y_geo',self.y)
            fdtd.putv('z_geo',self.z)
            fdtd.putv('eps_geo',full_eps)

            ## We delete and re-add the import to avoid a warning
            script=('select("import");'
                    'delete;'
                    'addimport;'
                    'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);')
            fdtd.eval(script)
