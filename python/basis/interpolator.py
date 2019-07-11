# @author Hillebrand, Fabian
# @date   2019

import scipy as sp

################################################################################
class Interpolator:
  """!
    @brief Provides an interpolator for a regular grid with consistent stepsize
           along an axis.

    The interpolation scheme used is currently piecewise linear polynomials.
    As such, the convergence rate is algebraic with a rate of 2 and a rate of
    1 for the derivative of the interpolation.
    Note that higher derivatives cannot be achieved with linear poylnomials.

    The method may be subject to change, though it is tedious as we are in 3 
    dimensions!

    @attention No care is taken for periodicity or out of bound: Make sure all
               points to be interpolated are within the regular grid!
  """

  def __init__(self, x, f):
    """!
      @param x Grid axes.
      @param f Function evaluated on complete grid.
    """
    self.x = x
    self.f = f
    self.dx = x[0][1]-x[0][0]
    self.dy = x[1][1]-x[1][0]
    self.dz = x[2][1]-x[2][0]

  ##############################################################################
  def __call__(self, x, y, z):
    """!
      @brief Evaluates interpolated value at given points.

      @param x, y, z Positions.
    """
    indX = ((x-self.x[0][0])/self.dx).astype(int)
    indY = ((y-self.x[1][0])/self.dy).astype(int)
    indZ = ((z-self.x[2][0])/self.dz).astype(int)

    return ((self.x[0][indX+1]-x)*(                                 \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1]))   \
           +(x-self.x[0][indX])*(                                   \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])))\
      / (self.dx*self.dy*self.dz)

  ##############################################################################
  def gradient(self, x, y, z, direct):
    """!
      @brief Evaluates gradient of interpolation in specified direciton.

      @param x, y, z Position.
      @param direct  Direction of gradient (x=1,y=2,z=3).
    """
    indX = ((x-self.x[0][0])/self.dx).astype(int)
    indY = ((y-self.x[1][0])/self.dy).astype(int)
    indZ = ((z-self.x[2][0])/self.dz).astype(int)

    # The lazy solution: copy-paste + adjust (could do something
    # fancy with indices but whether it's really worth the effort.)
    if direct == 1:
      return ((                                                     \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1])))  \
        / (self.dx*self.dy*self.dz)

    if direct == 2:
      return ((                                                     \
              (self.x[0][indX+1]-x)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY+1,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX,indY+1,indZ+1])    \
             +(x-self.x[0][indX])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY+1,indZ]   \
                +(z-self.x[2][indZ])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[0][indX+1]-x)*(                               \
                 (self.x[2][indZ+1]-z)*self.f[indX,indY,indZ]       \
                +(z-self.x[2][indZ])*self.f[indX,indY,indZ+1])      \
             +(x-self.x[0][indX])*(                                 \
                 (self.x[2][indZ+1]-z)*self.f[indX+1,indY,indZ]     \
                +(z-self.x[2][indZ])*self.f[indX+1,indY,indZ+1])))  \
        / (self.dx*self.dy*self.dz)
    if direct == 3: 
      return ((                                                     \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[0][indX+1]-x)*self.f[indX,indY,indZ+1]     \
                +(x-self.x[0][indX])*self.f[indX+1,indY,indZ+1])    \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[0][indX+1]-x)*self.f[indX,indY+1,indZ+1]   \
                +(x-self.x[0][indX])*self.f[indX+1,indY+1,indZ+1])) \
            -(                                                      \
              (self.x[1][indY+1]-y)*(                               \
                 (self.x[0][indX+1]-x)*self.f[indX,indY,indZ]       \
                +(x-self.x[0][indX])*self.f[indX+1,indY,indZ])      \
             +(y-self.x[1][indY])*(                                 \
                 (self.x[0][indX+1]-x)*self.f[indX,indY+1,indZ]     \
                +(x-self.x[0][indX])*self.f[indX+1,indY+1,indZ])))  \
        / (self.dx*self.dy*self.dz)
    raise NotImplementedError( \
      "Gradient in direction {} is not available".format(direct))

################################################################################
