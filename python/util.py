# @author Hillebrand, Fabian
# @date   2018-2019

import numpy as np

# TODO fix this: does not completely symmetrizes coefficients
def symmetrizeCoeffs(coeffs, eigs):
  """!
    @brief Symmetrizes coefficients from Chen's derivative rule.

    Symmetrization is simply obtained by doubling the entries
    swapping coefficients belonging to px- and py-orbital.

    @param coeffs Chen's derivative rule coefficients.
    @param eigs   Eigenvalues belonging to coefficients.

    @return Pair containg new coefficients and new eigenvalues.
  """
  # Number of spins
  noSpins = len(eigs)
  eigsNew = [None]*noSpins
  coeffsNew = [None]*noSpins
  for spinIdx in range(noSpins):
    eigsNew[spinIdx] = np.append(eigs[spinIdx], eigs[spinIdx])
    coeffsNew[spinIdx] = np.append(coeffs[spinIdx], \
      coeffs[spinIdx][:,[0,3,2,1]].copy(), 0)
  return coeffsNew, eigsNew


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


def getHeightIndices(heights, lVec, dimGrid, atoms):
  """!
    @brief Restricts given heights to heights within a grid.

    @param heights Heights that are searched after.
    @param lVec    lVec that describes the grid.
    @param dimGrid Dimension of grid.
    @param atoms   Sample atoms (heights are with respect to this).

    @return True heights that have been achieved in grid.
            Indices of the heights in the grid.
  """
  # Top most atom of sample
  topZ = np.max(atoms.positions[:,2])
  # Step size in grid
  dz = lVec[3,2] / dimGrid[2]

  heightIds = np.array([int(height / dz) for height in heights])
  trueHeights = heightIds*dz

  return trueHeights, heightIds
