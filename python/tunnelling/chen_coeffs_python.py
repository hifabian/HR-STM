# @author Hillebrand, Fabian
# @date   2018-2019

from . import chen_coeffs_abc

import time

import math
import copy as cp
import numpy as np
import scipy.sparse as sp

################################################################################
class ChenCoeffsPython(chen_coeffs_abc.ChenCoeffsAbstract):
  """!
    @brief Structure that provides access to coefficients for each point in a
           grid including associated eigenvalues.

    The coefficients can be rotated for the grid. The rotation is given through
    two different grids corresponding to the points of rotation and the rotated
    points. The axis of rotation is fixed in the x-y-plane.

    This structure provides access to the coefficients via bracket operators.
    The following structure is provided: [spinIdx,eneIdx][orbIdx][X,Y,Z].
    The orbIdx corresponds to {s, py, pz, px}-orbitals.

    @attention The evaluation is lazy and only done when accessed. Avoid
               accessing too much. The lazy evaluation is restricted to a
               sparse matrix-vector multiplication.
  """

  ##############################################################################
  ## OVERWRITTEN METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ## Member methods
  def __getitem__(self, idxTuple):
    """!
      @brief Provides bracket-operator access.

      In particular, this method invokes a computation on the grid. This is
      done here to save memory at the cost of potential recomputation.

      @param idxTuple Tupel of indices [tunnelIdx, spinIdx, eigIdx]

      @return Coefficient for Chen's derivative rule for each grid point.
    """
    # Unpack indices
    tunnelIdx, spinIdx, eigIdx = idxTuple
    if self._coeffs is not None \
      and self._eigC == eigIdx \
      and self._spinC == spinIdx \
      and self._tunnelC == tunnelIdx:
      return self._coeffs
    # Storage container
    coeffs = np.empty(((self.noOrbs+1)**2,)+self.dimGrid)

    # s-orbitals: Are never rotated
    coeffs[0].fill(self._singles[tunnelIdx][spinIdx][eigIdx,0])
    if self.noOrbs > 0:
      # p-orbitals: Are rotated like vectors
      coeffs[1].fill(self._singles[tunnelIdx][spinIdx][eigIdx,1])
      coeffs[2].fill(self._singles[tunnelIdx][spinIdx][eigIdx,2])
      coeffs[3].fill(self._singles[tunnelIdx][spinIdx][eigIdx,3])
      # Flat view of coeffs[1:4]
      flatCoeffs = coeffs[1:4].ravel()
      # Provoke write into flat view instead of overwriting variable with [:]
      flatCoeffs[:] = self._rotMatrix[tunnelIdx]*coeffs[1:4].flatten()

    # Save some information
    self._coeffs = coeffs
    self._tunnelC, self._spinC, self._eigC = idxTuple
    return coeffs

  def setGrids(self, grids):
#    self._grids = cp.deepcopy(grids)
    self._grids = grids

    # s-orbtial on tip only
    if self.noOrbs == 0:
      return

    start = time.time()
    # p-orbital on tip
    self._rotMatrix = [None]*self.noTunnels
    if not self._rotate:
      for tunnelIdx in range(self.noTunnels):
        self._rotMatrix[tunnelIdx] = 1
    else:
      noPoints = np.prod(self.dimGrid)
      shiftedGrids = []
      data = []
      for i in range(self.noTunnels):
        shiftedGrids.append(
          np.array(
          [grids[i+1][0]-grids[i][0],
           grids[i+1][1]-grids[i][1],
           grids[i+1][2]-grids[i][2]]
          ).reshape(3,noPoints))
        data.append(np.empty(9*noPoints))
      # Sparse matrix storage as COO
      rowIdx = np.empty(9*noPoints, dtype=int)
      colIdx = np.empty(9*noPoints, dtype=int)

      idxHelper = np.arange(3*noPoints,dtype=int)
      for tunnelIdx in range(self.noTunnels):
        v = np.array([0.0,0.0,-1.0])
        # Rotated vector
        w = shiftedGrids[tunnelIdx]
        w /= np.linalg.norm(w,axis=0)
        # Rotation axis (no rotation around x-y)
        n = np.cross(v,w,axisb=0).transpose()
        # Trigonometric values
        cosa = np.dot(v,w)
        sina = (1-cosa**2)**0.5

        #  R = [[n[0]**2*(1-cosa)+cosa, n[0]*n[1]*(1-cosa)-n[2]*sina, n[0]*n[2]*(1-cosa)+n[1]*sina],
        #       [n[1]*n[0]*(1-cosa)+n[2]*sina, n[1]**2*(1-cosa)+cosa, n[1]*n[2]*(1-cosa)-n[0]*sina],
        #       [n[2]*n[1]*(1-cosa)-n[1]*sina, n[2]*n[1]*(1-cosa)+n[0]*sina, n[2]**2*(1-cosa)+cosa]]
        # Permutation matrix for betas: (y,z,x) -> (x,y,z)
        #  P = [[0,0,1],
        #       [1,0,0],
        #       [0,1,0]]
        # Note: The rotational matrix R is with respect to the sample which is R^T for the tip
        #       --> R to R^T for going from tip to sample, R^T to R for gradient
        data[tunnelIdx][:noPoints]             = n[1]**2*(1-cosa)+cosa
        data[tunnelIdx][noPoints:2*noPoints]   = n[2]*n[1]*(1-cosa)-n[0]*sina
        data[tunnelIdx][2*noPoints:3*noPoints] = n[0]*n[1]*(1-cosa)+n[2]*sina
        data[tunnelIdx][3*noPoints:4*noPoints] = n[1]*n[2]*(1-cosa)+n[0]*sina
        data[tunnelIdx][4*noPoints:5*noPoints] = n[2]**2*(1-cosa)+cosa
        data[tunnelIdx][5*noPoints:6*noPoints] = n[0]*n[2]*(1-cosa)-n[1]*sina
        data[tunnelIdx][6*noPoints:7*noPoints] = n[1]*n[0]*(1-cosa)-n[2]*sina
        data[tunnelIdx][7*noPoints:8*noPoints] = n[2]*n[0]*(1-cosa)+n[1]*sina
        data[tunnelIdx][8*noPoints:9*noPoints] = n[0]**2*(1-cosa)+cosa
        # rowIdx = [0,1,2,...,0,1,2,...,0,1,2,...]
        rowIdx = np.tile(idxHelper,3)
        # colIdx = [0,0,0,1,1,1,2,2,2,...]
        colIdx = np.repeat(idxHelper,3)
      del idxHelper

      # Build large matrices
      for tunnelIdx in range(self.noTunnels):
        self._rotMatrix[tunnelIdx] = \
          sp.csr_matrix(sp.coo_matrix((data[tunnelIdx],(rowIdx,colIdx)), \
                                       shape=(3*noPoints, 3*noPoints)))

    end = time.time()
    print("Rotational matrices took {} seconds".format(end-start))

################################################################################
