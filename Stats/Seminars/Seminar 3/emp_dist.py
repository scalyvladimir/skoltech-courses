import numpy as np


class EmpiricalDistribution:
    """
    This class provides empirical distribution functionality. It can be used to 
    get point masses and cdf arrays for plotting.
    
    This implementations assumes that
    F(x) = P(X <= x) = Sum_i I[X_i <= x]
    """
    def __init__(self, samples):
        """
        Initialize empirical distribution.
        
        Arguments:
        - samples: numpy array, list or tuple of 1D samples
        """
        self.samples = np.sort(samples)
        
    def get_pdf_data(self):
        """
        Get data required to plot PDF. 
        """
        points, counts = np.unique(self.samples, return_counts=True)
        density = counts / counts.sum()
        return points, density
    
    def get_continuous_pdf_data_(self, n_points=10, covariance_factor=None):
        """
        Get data required to plot PDF. 
        """
        points, weights = self.get_pdf_data()
        density = stats.gaussian_kde(points, weights=weights)
        if covariance_factor is not None:
            density.covariance_factor = lambda : covariance_factor
            density._compute_covariance()
        
        xs = np.linspace(points[0], points[-1], n_points)
        return xs, density(xs)
    
    def get_cdf_data(self, append_borders=1e-3):
        """
        Get data required to plot CDF.
        
        Arguments:
        - append_borders: nonegative number. 
            If append_borders > 0, adds border points x_left and x_right such that F(x_left) = 0, F(x_right) = 1.
            It is required for appropriate plotting since otherwise the plots do not show levels 0 and 1 of CDF. 
            The values of x_left and x_right are found as follows:
                x_left = min(samples) - append_borders * (max(samples) - min(samples))
                x_right = max(samples) +  append_borders * (max(samples) - min(samples))
            
        Returns shape (points, cumulatives):
        - points: numpy array which contains stored samples (and border points if append_borders > 0) 
            in ascending order.
        - cumulatives: numpy array which contains CDF values for the returned points. 

        """
        # Your code here
        
        points, densities = self.get_pdf_data()
        cumulatives = np.cumsum(densities)

        x_delta = points[-1] - points[0]
        x_left = points[0] - append_borders * x_delta
        x_right = points[-1] +  append_borders * x_delta

        return np.r_[x_left, points, x_right], np.r_[0, cumulatives, 1]
