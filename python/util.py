# @author Hillebrand, Fabian
# @date   2019

import numpy as np

# ==============================================================================
# ------------------------------------------------------------------------------

# ==============================================================================

def apply_bounds(grid, lVec):
  """!
    @brief Restrict grids to box in x- and y-direction.

    @param grid Grid to be restricted.
    @param lVec Information on box.

    @return Grid with position restricted to periodic box.
  """
  dx = lVec[1,0]-lVec[0,0]
  grid[0][grid[0] >= lVec[1,0]] -= dx
  grid[0][grid[0] < lVec[0,0]]  += dx

  dy = lVec[2,1]-lVec[0,1]
  grid[1][grid[1] >= lVec[2,1]] -= dy
  grid[1][grid[1] < lVec[0,1]]  += dy

  return grid


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================


def constCoeffs(eMin, eMax, de=0.1, s=0.0, py=0.0, pz=0.0, px=0.0):
  """!
    @brief Creates coefficients for seperated tunneling to each orbital.
  """
  nE = int((eMax-eMin) / de)+1
  cc = np.array([s, py, pz, px]) != 0.0
  coeffs = np.empty((nE*sum(cc),4),)
  eigs = np.empty(nE*sum(cc))
  for n in range(nE):
    idx = n*(sum(cc))
    if s != 0.0:
      coeffs[idx+sum(cc[:1])-1,:] = [s**0.5, 0.0, 0.0, 0.0]
    if py != 0.0:
      coeffs[idx+sum(cc[:2])-1,:] = [0.0, py**0.5, 0.0, 0.0]
    if pz != 0.0:
      coeffs[idx+sum(cc[:3])-1,:] = [0.0, 0.0, pz**0.5, 0.0]
    if px != 0.0:
      coeffs[idx+sum(cc[:4])-1,:] = [0.0, 0.0, 0.0, px**0.5]
    eigs[idx:idx+sum(cc)] = eMin+de*n

  return [coeffs], [eigs]


# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
