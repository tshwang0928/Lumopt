import numpy as np

class Wavelengths(object):
    
    def __init__(self, start , stop = None, points = 1): 
        self.custom = len( np.array(start).flatten() ) > 1 
        if self.custom:
            if stop is not None:
                raise UserWarning('If the first argument to Wavelength is an array, it must be the only argument.')
            self.wavelength_points = np.array(start).flatten()
            self.wavelength_points.sort()
            self.start = None
            self.points = len(self.wavelength_points)
        else:
            self.start = float(start) 
            self.stop = float(stop) if stop else start
            if self.stop < self.start:
                raise UserWarning('span must be positive.')
            self.points = int(points) # if not self.custom else len(self.wavelength_points)
            if self.points < 1:
                raise UserWarning('number of points must be positive.')
            if self.stop == self.start and self.points > 1:
                raise UserWarning('zero length span with multiple points.')
            
    def min(self):
        if not self.custom:
            return float(self.start)  
        else:
            return float(self.wavelength_points[0])

    def max(self):
        if not self.custom:
            return float(self.stop)
        else:
            return float(self.wavelength_points[-1])

    def __len__(self):
        return int(self.points)

    def __getitem__(self, item):
        return self.asarray()[item]

    def asarray(self):
        if not self.custom:
            return np.linspace(start = self.start, stop = self.stop, num = self.points) if self.points > 1 else 0.5*np.array([self.start + self.stop])
        else:
            return self.wavelength_points


