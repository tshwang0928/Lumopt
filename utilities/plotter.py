""" Copyright chriskeraly
    Copyright (c) 2019 Lumerical Inc. """

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FileMovieWriter
from matplotlib.cbook import flatten

class SnapShots(FileMovieWriter):
    ''' Grabs the image information from the figure and saves it as a movie frame. '''

    supported_formats = ['png', 'jpeg', 'pdf']

    def __init__(self, *args, extra_args = None, **kwargs):
        super().__init__(*args, extra_args = (), **kwargs) # stop None from being passed

    def setup(self, fig, dpi, frame_prefix):
        super().setup(fig, dpi, frame_prefix, clear_temp = False)
        self.fname_format_str = '%s%%d.%s'
        self.temp_prefix, self.frame_format = self.outfile.rsplit('.', 1)

    def grab_frame(self, **fig_kwargs):
        ''' All keyword arguments in fig_kwargs are passed on to the 'savefig' command that saves the figure. '''
        with self._frame_sink() as myframesink:
            self.fig.savefig(myframesink, format = self.frame_format, dpi = self.dpi, **fig_kwargs)

    def finish(self):
        self._frame_sink().close()

class Plotter(object):
    '''
        Orchestrates the generation of plots during the optimization.

        Parameters
        ----------
        :param movie:            Indicates if the evolution of parameters should be recorded as a movie
        :param plot_history      Indicates if we should plot the history of the parameters and gradients. Should
                                 be set to False for large (e.g. >100) numbers of parameters 
        :param plot_fields       Indicates if we should plot the field and gradient information. Default is True
                                 but for larger 3d optimizations it can make sense todisable the plotting to reduce
                                 memory consumption and to improve performance.
    '''

    def __init__(self, movie = True, plot_history = True, plot_fields = True):
        self.plot_history = plot_history
        self.plot_fields = plot_fields

        if self.plot_fields:
            if plot_history:
                self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
            else:
                self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
        else:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        ## Flatten the axes because it is difficult to keep track otherwise:
        l = list(map(list, zip(*self.ax)))
        self.ax = list(flatten(l))

        self.fig.show()
        self.movie = movie
        if movie:
            metadata = dict(title = 'Optimization', artist='lumopt', comment = 'Continuous adjoint optimization')
            self.writer = SnapShots(fps = 2, metadata = metadata)

    def clear(self):
        for x in self.ax:
            x.clear()

    def update_fom(self, optimization):
        if self.plot_fields and self.plot_history:
            optimization.plot_fom(fomax=self.ax[0], paramsax=self.ax[4], gradients_ax=self.ax[5])
        else:
            optimization.plot_fom(fomax=self.ax[0], paramsax=None, gradients_ax=None)
        
    def update_gradient(self, optimization):
        if self.plot_fields and hasattr(optimization, 'plot_gradient'):
            optimization.plot_gradient(self.fig, self.ax[3], self.ax[2])

    def update_geometry(self, optimization):
        if not optimization.geometry.plot(self.ax[1]):
            if hasattr(optimization, 'gradient_fields'):
                optimization.gradient_fields.plot_eps(self.ax[1])
                
    def draw_and_save(self):
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.movie:
            self.writer.grab_frame()
            print('Saved frame')

    def set_legend(self, legend):
        self.ax[0].legend(legend)

