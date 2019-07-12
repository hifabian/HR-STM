# @author Hillebrand, Fabian
# @date   2019

import scipy as sp
import numpy as np

################################################################################
class Interpolator:
  """!
    @brief Provides an interpolator for a regular grid with consistent stepsize
           along an axis.

    The interpolation scheme used is currently piecewise linear polynomials.
    As such, the convergence rate is algebraic with a rate of 2.
    First derivatives are achieved using second order finite differences to not
    stump the convergence rate.

    @attention No care is taken for periodicity or out of bound: Make sure all
               points to be interpolated are within the regular grid!
  """

  def __init__(self, x, f):
    """!
      @param x Grid axes.
      @param f Function evaluated on complete grid.
    """
    self.intrp = sp.interpolate.RegularGridInterpolator(self.x, self.f, \
      method='linear')
    self.intrX = sp.interpolate.RegularGridInterpolator(self.x, \
      np.gradient(self.f,self.x[0],edge_order=2,axis=0), \
      method='linear')
    self.intrY = sp.interpolate.RegularGridInterpolator(self.x, \
      np.gradient(self.f,self.x[1],edge_order=2,axis=1), \
      method='linear')
    self.intrZ = sp.interpolate.RegularGridInterpolator(self.x, \
      np.gradient(self.f,self.x[2],edge_order=2,axis=2), \
      method='linear')

  ##############################################################################
  def __call__(self, x, y, z):
    """!
      @brief Evaluates interpolated value at given points.

      @param x, y, z Positions.
    """
    self.intrp(np.array([x,y,z]))

  ##############################################################################
  def gradient(self, x, y, z, direct):
    """!
      @brief Evaluates gradient of interpolation in specified direciton.

      @param x, y, z Position.
      @param direct  Direction of gradient (x=1,y=2,z=3).
    """
    if direct == 1:
      return self.intrX(np.array([x,y,z]))
    if direct == 2:
      return self.intrY(np.array([x,y,z]))
    if direct == 3:
      return self.intrZ(np.array([x,y,z]))
    raise NotImplementedError( \
      "Gradient in direction {} is not available".format(direct))


################################################################################
