# @author Hillebrand, Fabian
# @date   2018-2019

from . import wavefunction_abc

import time

import ctypes
import os
import numpy as np
import sysconfig as sc


################################################################################
class WavefunctionCPP(wavefunction_abc.WavefunctionAbstract):
  """!
    @brief Realization of abstract wavefunction class using exponentials as
           basis set as is done in PPSTM code. Evaluation is done in C++.

    Currently the basis function supports s- and p-orbitals for the tip and
    s- and p-orbitals for the sample (any combination thereof).

    @attention Grid must be flattened to shape (noPoints,3).
  """

  def __init__(self, **kwargs):
    """!
      @brief Initializer.

      @param workfunction Value for workfunction.
      @param rcut         Cutoff radius used in calculations.
      @param pbc          3-tupel of booleans indicating periodicity.
      @param abc          Size of box for atoms used by periodicity.
      @param See WavefunctionAbstract
    """
    for spinIdx in range(len(kwargs['eigs'])):
      if not kwargs['coefficients'][spinIdx].flags['C_CONTIGUOUS'] \
        or not kwargs['coefficients'][spinIdx].flags['ALIGNED']:
        raise ValueError("Coefficients are not C-contiguous or aligned.")
      if not kwargs['eigs'][spinIdx].flags['C_CONTIGUOUS'] \
        or not kwargs['eigs'][spinIdx].flags['ALIGNED']:
        raise ValueError("Eigenvalues are not C-contiguous or aligned.")
    if not kwargs['atoms'].get_positions().flags['C_CONTIGUOUS'] \
      or not kwargs['atoms'].get_positions().flags['ALIGNED']:
      raise ValueError("Atoms are not C-contiguous or aligned.")

    super().__init__(**kwargs)
    self._workfunction = kwargs['workfunction']
    self._noOrbsSample = int(np.shape(kwargs['coefficients'][0])[-1]**0.5-1)
    self.rcut = kwargs['rcut']
    self.pbc = np.array(kwargs['pbc'], dtype=np.bool_)
    self.abc = np.array(kwargs['abc'], dtype=np.double)
    self._computed = False
  
  ##############################################################################
  ## Static methods
  @staticmethod
  def compile():
    """! @brief Compiles the C++ code. """
    includes = "-I"+sc.get_paths()["include"] + " -I"+np.get_include()
    path = os.path.dirname(os.path.abspath(__file__))
    os.system("make --directory "+path+"/../../cpp INCLUDES='"+includes+"'")

  ##############################################################################
  ## Member methods
  def _compute(self):
    """! @brief Computes the wavefunctions. """
    array1d = np.ctypeslib.ndpointer(dtype=np.double,  ndim=1, flags='CONTIGUOUS')
    array1b = np.ctypeslib.ndpointer(dtype=np.bool_,  ndim=1, flags='CONTIGUOUS')

    # Path to this file
    path = os.path.dirname(os.path.abspath(__file__))
    # C++ function
    clib = ctypes.CDLL(path+"/../../cpp/libwfn.so")
    clib.computeWFN.argtypes = [ \
      # workfunction, rCut, pbc, abc
      ctypes.c_double, ctypes.c_double, array1b, array1d, \
      # coeffsObject, eigsObject, atoms, grid
      ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object, \
      # wfnObject
      ctypes.py_object]

    # Total number of grid points
    noPoints = np.prod(self.dimGrid)
    noDer = (self.noOrbsSample+1)**2
    # Storage container
    self._wfn = []

    start = time.time()
    for gridIdx in range(self.noGrids):
      self._wfn.append([np.zeros((self.noEigs[spinIdx],(self.noOrbsTip+1)**2,) \
        +self.dimGrid)
        for spinIdx in range(self.noSpins)])
      for spinIdx in range(self.noSpins):
        clib.computeWFN(self._workfunction, \
          self.rcut, self.pbc, self.abc, \
          self._coefficients[spinIdx], self._eigs[spinIdx], \
          self._atoms.get_positions(), \
          self._grids[gridIdx], self._wfn[gridIdx][spinIdx])
    end = time.time()
    print("Wavefunction took {} seconds".format(end-start))
    self._computed = True

  ##############################################################################
  ## Member variables
  @property
  def noOrbsSample(self):
    return self._noOrbsSample
  @property
  def wn(self):
    return self._workfunction
  @property
  def workfunction(self):
    return self._workfunction

  ##############################################################################
  ## OVERWRITTEN METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ## Member methods
  def setGrids(self, grids):
    """! @attention Grid must be flattened to shape (noPoints,3). """
    self._grids = grids
    if not self._computed:
      self._compute()

################################################################################
