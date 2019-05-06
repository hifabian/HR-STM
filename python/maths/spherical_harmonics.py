# @author Hillebrand, Fabian
# @date   2018-2019

import numpy as np

"""!
  This file provides real spherical harmonics and their derivatives.
"""

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def spherical_harmonic(l, m, grid):
  """!
    @brief Evaluates real spherical harmonics multiplied with r^l for values on
           a Cartesian grid. Normalization is unknown.

    @param l    Degree of spherical harmonic (s-, p-, or d-orbitals).
    @param m    Order of spherical harmonic (in [-l, l], see CP2K convention).
    @param grid Grid values.

    @return A grid containing the values of the real spherical harmonics 
            multiplied with r^l

    @note Taken from Kristjan Eimre and adjusted for needs.
          See: https://github.com/eimrek/atomistic_tools
  """

  # These look like combinations of other spherical harmonics to obtain
  # real spherical harmnics. Also, Chen has a table with the derivative
  # rules for these!
  # Problem: Are the coefficients actually the correct ones? Kristjan uses
  # this but I am not sure if that really means anything

  const = (2./np.pi)**(3./4.)

  # s orbtial
  if (l, m) == (0, 0):
    return const

  # p orbitals
  if (l, m) == (1, -1):
    return 2.*const*grid[:,:,:,1]
  if (l, m) == (1, 0):
    return 2.*const*grid[:,:,:,2]
  if (l, m) == (1, 1):
    return 2.*const*grid[:,:,:,0]

  # d orbitals
  if (l, m) == (2, -2):
    return 4.*const*grid[:,:,:,0]*grid[:,:,:,1]
  if (l, m) == (2, -1):
    return 4.*const*grid[:,:,:,1]*grid[:,:,:,2]
  if (l, m) == (2, 0):
    return 2.*const/np.sqrt(3.)*(2.*grid[:,:,:,2]**2-grid[:,:,:,0]**2-grid[:,:,:,1]**2)
  if (l, m) == (2, 1):
    return 4.*const*grid[:,:,:,0]*grid[:,:,:,2]
  if (l, m) == (2, 2):
    return 2.*const*(grid[:,:,:,0]**2-grid[:,:,:,1]**2)
    
  raise NotImplementedError("Degree higher than l > 2 not supported (f-orbtial and onward).")

# ------------------------------------------------------------------------------

def spherical_harmonic_d0(l, m, grid):

  const = (2./np.pi)**(3./4.)

  # s orbtial
  if (l, m) == (0, 0):
    return 0

  # p orbitals
  if (l, m) == (1, -1):
    return 0
  if (l, m) == (1, 0):
    return 0
  if (l, m) == (1, 1):
    return 2.*const*grid[:,:,:,0]

  # d orbitals
  if (l, m) == (2, -2):
    return 4.*const*grid[:,:,:,1]
  if (l, m) == (2, -1):
    return 0
  if (l, m) == (2, 0):
    return 2.*const/np.sqrt(3.)*(-2*grid[:,:,:,0])
  if (l, m) == (2, 1):
    return 4.*const*grid[:,:,:,2]
  if (l, m) == (2, 2):
    return 2.*const*(2*grid[:,:,:,0])
    
  raise NotImplementedError("Degree higher than l > 2 not supported (f-orbtial and onward).")
# ------------------------------------------------------------------------------

def spherical_harmonic_d1(l, m, grid):

  const = (2./np.pi)**(3./4.)

  # s orbtial
  if (l, m) == (0, 0):
    return 0

  # p orbitals
  if (l, m) == (1, -1):
    return 2.*const*grid[:,:,:,1]
  if (l, m) == (1, 0):
    return 0
  if (l, m) == (1, 1):
    return 0

  # d orbitals
  if (l, m) == (2, -2):
    return 4.*const*grid[:,:,:,0]
  if (l, m) == (2, -1):
    return 4.*const*grid[:,:,:,2]
  if (l, m) == (2, 0):
    return 2.*const/np.sqrt(3.)*(-2*grid[:,:,:,1])
  if (l, m) == (2, 1):
    return 0
  if (l, m) == (2, 2):
    return 2.*const*(-2*grid[:,:,:,1])
    
  raise NotImplementedError("Degree higher than l > 2 not supported (f-orbtial and onward).")

# ------------------------------------------------------------------------------

def spherical_harmonic_d2(l, m, grid):

  const = (2./np.pi)**(3./4.)

  # s orbtial
  if (l, m) == (0, 0):
    return 0

  # p orbitals
  if (l, m) == (1, -1):
    return 0
  if (l, m) == (1, 0):
    return 2.*const*grid[2]
  if (l, m) == (1, 1):
    return 0

  # d orbitals
  if (l, m) == (2, -2):
    return 0
  if (l, m) == (2, -1):
    return 4.*const*grid[:,:,:,1]
  if (l, m) == (2, 0):
    return 2.*const/np.sqrt(3.)*(4.*grid[:,:,:,2])
  if (l, m) == (2, 1):
    return 4.*const*grid[:,:,:,0]
  if (l, m) == (2, 2):
    return 0
    
  raise NotImplementedError("Degree higher than l > 2 not supported (f-orbtial and onward).")


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
