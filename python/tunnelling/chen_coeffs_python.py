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
    self._grids = cp.deepcopy(grids)

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
        shiftedGrids.append(np.array(grids[i+1]-grids[i], \
          order='C').reshape(noPoints,3))
        data.append(np.empty(9*noPoints))
      # Sparse matrix storage as COO
      rowIdx = np.empty(9*noPoints, dtype=int)
      colIdx = np.empty(9*noPoints, dtype=int)

      curIdx = 0 # Current entry for data
      for pointIdx in range(noPoints):
        # Rotation matrix is defined by:
        #   R*x = n*(n . x) + cos(a)*(n ^ x) + sin(a)*(n ^ x) ^ n
        # with n the axis of rotation and a the angle. The first term 
        # corresponds to the unchanged distance along the rotation axis.
        # The second term is the height, orthogonal to x, while the third
        # is the distance to the axis with respect to a plane.
        # One may write it R as a matrix explicitely:
        #  R = np.array([[n[0]**2*(1-cosa)+cosa, n[0]*n[1]*(1-cosa)-n[2]*sina, n[0]*n[2]*(1-cosa)+n[1]*sina],
        #                [n[1]*n[0]*(1-cosa)+n[2]*sina, n[1]**2*(1-cosa)+cosa, n[1]*n[2]*(1-cosa)-n[0]*sina],
        #                [n[2]*n[1]*(1-cosa)-n[1]*sina, n[2]*n[1]*(1-cosa)+n[0]*sina, n[2]**2*(1-cosa)+cosa]])
        # Permutation matrix for betas: (y,z,x) -> (x,y,z)
        #  P = np.array([[0,0,1],
        #                [1,0,0],
        #                [0,1,0]])
        # Note: The rotational matrix R is with respect to the sample which is R^T for the tip
        #       --> R to R^T for going from tip to sample, R^T to R for gradient

        # Flattened indices for p-orbitals
        pxIdx = pointIdx
        pyIdx = pointIdx+noPoints
        pzIdx = pointIdx+2*noPoints

        # Indices for large matrix
        rowIdx[curIdx:curIdx+9] = [pxIdx, pxIdx, pxIdx, pyIdx, pyIdx, pyIdx, pzIdx, pzIdx, pzIdx]
        colIdx[curIdx:curIdx+9] = [pxIdx, pyIdx, pzIdx, pxIdx, pyIdx, pzIdx, pxIdx, pyIdx, pzIdx]

        for tunnelIdx in range(self.noTunnels):
          # Reference vector
          v = np.array([0.0, 0.0, -1.0])
          v /= np.linalg.norm(v)
          # Rotated vector
          w = shiftedGrids[tunnelIdx][pointIdx]
          w /= np.linalg.norm(w)
          # Rotation axis (no rotation around x-y)
          n = np.cross(v,w)
          # Check if actually rotated
          if not math.isclose(np.linalg.norm(n), 0.0):
            n /= np.linalg.norm(n)
            # Trigonometric values for rotation angle
            cosa = np.dot(v,w)
            sina = (1-cosa*cosa)**0.5
            # Resulting matrix for p-orbitals: P^T * R^T * P
            PtRtP = np.array([[n[1]**2*(1-cosa)+cosa, n[2]*n[1]*(1-cosa)-n[0]*sina, n[0]*n[1]*(1-cosa)+n[2]*sina], \
                              [n[1]*n[2]*(1-cosa)+n[0]*sina, n[2]**2*(1-cosa)+cosa, n[0]*n[2]*(1-cosa)-n[1]*sina], \
                              [n[1]*n[0]*(1-cosa)-n[2]*sina, n[2]*n[0]*(1-cosa)+n[1]*sina, n[0]**2*(1-cosa)+cosa]])
          
          else:
            PtRtP = np.array([[1.0, 0.0, 0.0], \
                              [0.0, 1.0, 0.0], \
                              [0.0, 0.0, 1.0]])
          data[tunnelIdx][curIdx:curIdx+9] = PtRtP.ravel()
        curIdx += 9
      # Build large matrices
      for tunnelIdx in range(self.noTunnels):
        self._rotMatrix[tunnelIdx] = \
          sp.csr_matrix(sp.coo_matrix((data[tunnelIdx],(rowIdx,colIdx)), \
                                       shape=(3*noPoints, 3*noPoints)))

    end = time.time()
#    print("ChenCoeffs took {} seconds".format(end-start))

################################################################################
