# @author Hillebrand, Fabian
# @date   2018-2019

from . import chen_coeffs_abc

import time

import ctypes
import os
import numpy as np
import sysconfig as sc
import scipy.sparse as sp

################################################################################
class ChenCoeffsCPP(chen_coeffs_abc.ChenCoeffsAbstract):
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

    @attention Grid must be flattened to shape (noPoints,3).
  """

  def __init__(self, **kwargs):
    """!
      @brief Initializer

      @note Checks if inputs are suitable for C++.

      @param See ChenCoeffsAbstract.
    """
    for spinIdx in range(len(kwargs['eigs'])):
      for tunnelIdx in range(len(kwargs['singles'])):
        if not kwargs['singles'][tunnelIdx][spinIdx].flags['C_CONTIGUOUS'] \
        or not kwargs['singles'][tunnelIdx][spinIdx].flags['ALIGNED']:
          raise ValueError("Singles are not C-contiguous or aligned.")
      if not kwargs['eigs'][spinIdx].flags['C_CONTIGUOUS'] \
        or not kwargs['eigs'][spinIdx].flags['ALIGNED']:
        raise ValueError("Eigenvalues are not C-contiguous or aligned.")

    super().__init__(**kwargs)

  ##############################################################################
  ## Static methods
  @staticmethod
  def compile():
    """!
      @brief Compiles the C++ code.
    """
    includes = "-I"+sc.get_paths()["include"] + " -I"+np.get_include()
    path = os.path.dirname(os.path.abspath(__file__)) 
    os.system("make --directory "+path+"/../../cpp INCLUDES='"+includes+"'")

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
      # Multiply rotational matrices (nested)
      tmp = self._rotMatrix[tunnelIdx]*coeffs[1:4].flatten()
#      for idx in range(tunnelIdx):
#        tmp = self._rotMatrix[tunnelIdx-idx]*tmp
      # Provoke write into flat view instead of overwriting variable with [:]
      flatCoeffs[:] = tmp

    # Save some information
    self._coeffs = coeffs
    self._tunnelC, self._spinC, self._eigC = idxTuple
    return coeffs

  def setGrids(self, grids):
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
      # Path to this file
      path = os.path.dirname(os.path.abspath(__file__))
      # C++ library
      clib = ctypes.CDLL(path+"/../../cpp/librot.so")
      # Define signature for C++ functions
      clib.computeRot.argtypes = [ \
        # shiftedGrid, rowIdx, colIdx, data
        ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object]

      noPoints = np.prod(self.dimGrid)
      shiftedGrids = []
      data = []
      for i in range(self.noTunnels):
        shiftedGrids.append(np.array(grids[i+1]-grids[i], order='C'))
        data.append(np.empty(9*noPoints))
      # Sparse matrix storage as COO
      rowIdx = np.empty(9*noPoints, dtype=np.intc)
      colIdx = np.empty(9*noPoints, dtype=np.intc)
      # Run C++ code  
      clib.computeRot(shiftedGrids,rowIdx,colIdx,data)
      # Build large sparse matrices
      for tunnelIdx in range(self.noTunnels):
        self._rotMatrix[tunnelIdx] = \
          sp.csr_matrix(sp.coo_matrix((data[tunnelIdx],(rowIdx,colIdx)), \
                                       shape=(3*noPoints, 3*noPoints)))
    end = time.time()
    print("ChenCoeffs took {} seconds".format(end-start))

################################################################################
