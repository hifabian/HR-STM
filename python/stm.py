# @author Hillebrand, Fabian
# @date   2018-2019

import time     # Debugging purposes
import os       # Used by plotting
import warnings # Used for warnings

import ctypes
import math

import copy as cp
from mpi4py import MPI
from collections import namedtuple

from basis.wavefunction_abc import *
from tunnelling.chen_coeffs_abc import *
from tunnelling.tunnel_matrix import *

################################################################################
class STM():
  """!
    @brief Class that runs a STM simulation.
  """

  def __init__(self, chenCoeff, wfn, atoms, grids, lVec, dosTypeSample, \
    dosArgsSample, comm, size, rank):
    """!
      @brief Initializer. Builds STM object but does not run it yet.

      The type of the density of states for the tip is assumed to be a dirac-
      function and should be given by a string for the sample. Supported are:
      - Lorentzian
      - Gaussian

      @param chenCoeff ChenCoeffs-object for derivative rule.
      @param wfn       Wavefunction-object for the sample.
      @param atoms     Atom-object for sample.
      @param grids     Tip positions as list of arrays. Proceeding entry serves
                       as reference for following entry. Only needed for process
                       with rank 0.
      @param lVec      Structure for scan.
      @param dosTypeSample String indicating type of density of states on
                           sample.
      @param dosArgsSample Arguments potentially needed for density of states
                           on sample.
      @param comm Communicator of MPI.
      @param size Number of processes.
      @param rank Rank of this process.
    """
    # MPI information
    self.comm = comm
    self.size = size
    self.rank = rank

    # Distribute grids from process with rank 0
    if self.rank == 0:
      # Check input grids
      for grid in grids:
        if not grid.flags['C_CONTIGUOUS'] or not grid.flags['ALIGNED']:
          raise ValueError("Grids are not C-contiguous or aligned.")
        

      # Values only needed by process with rank 0
      self.dimGrid = np.shape(grids[0])[:-1]
      self.lVec = lVec

      noGrids = len(grids)
      noPoints = self.dimGrid[0]*self.dimGrid[1]*self.dimGrid[2]

      # Length of the splits, needed for reassembly
      self.splitLengths = []
      for rankIdx in range(noPoints % self.size):
        self.splitLengths.append((noPoints // self.size + 1)*3)
      for rankIdx in range(self.size - (noPoints % self.size)):
        self.splitLengths.append((noPoints // self.size)*3)
      self.splitLengths = np.array(self.splitLengths)
      # Relative positions of data (viewed as C-like array)
      inputLengths = self.splitLengths
      offsetInput = np.insert(np.cumsum(inputLengths),0,0)[:-1]
    else:
      inputLengths = None
      offsetInput = None
      noGrids = None
      noPoints = None
    # Broadcasting information on splits
    inputLengths = self.comm.bcast(inputLengths, root=0)
    offsetInput = self.comm.bcast(offsetInput, root=0)
    noGrids = self.comm.bcast(noGrids, root=0)
    noPoints = self.comm.bcast(noPoints, root=0)
    # Scattered grids as localGrids
    if not self.rank == 0:
      # Dummy grids for other processes to make scattering easier
      grids = [None]*noGrids
    self.localGrids = []
    for gridIdx in range(noGrids):
      self.localGrids.append(np.empty((inputLengths[self.rank]//3,3)))
      self.comm.Scatterv([grids[gridIdx], inputLengths, \
        offsetInput, MPI.DOUBLE], self.localGrids[gridIdx], root=0)

    self.dimLocal = (np.shape(self.localGrids[0])[0],)
    self._atoms = atoms
    self._dosType = dosTypeSample
    self._dosSample = self._dos_type(dosTypeSample, dosArgsSample)

    self._chenCoeff = chenCoeff
    self._wfn = wfn

  ##############################################################################
  def _dos_type(self, dosType, dosArgs):
    """!
      @brief Determines which DoS is to be used.
    """
    if dosType == "Lorentzian":
      self._DosArgs = namedtuple('DosArgs', ['ene', 'eta'])
      self._DosArgs.__new__.__defaults__ = (None, dosArgs[0],)
      return self._dosLorentzian
    elif dosType == "Gaussian":
      self._DosArgs = namedtuple('DosArgs', ['ene', 'sigma'])
      self._DosArgs.__new__.__defaults__ = (None, dosArgs[0]/2.35482,)
      return self._dosGaussian
    else:
      raise NotImplementedError("Method "+dosType+" is not implemented.")

  def _dosConstant(self, args=[]):
    """!
      @brief Constant density of state. 
    """
    return 1.0

  def _dosLorentzian(self, args):
    """!
      @brief Lorentzian density of state.

      @args Arguments in form of a named tuple (ene, eta).
    """
    return 1./np.pi * args.eta / ( args.ene**2 + args.eta**2 )

  def _dosGaussian(self, args):
    """!
      @brief Gaussian density of state.

      @args Arguments in form of a named tuple (ene, sigma).
    """
    if abs(args.ene) > 4.710*args.sigma:
      return 0.0
    else:
      return np.exp(-(args.ene / args.sigma)**2) / (args.sigma*(2*np.pi)**0.5)

  ##############################################################################
  def _compute(self):
    """!
      @brief Evaluates the current for the processes. The current is stored
             in self.localCurrent and needs to be assembled.
    """
    self._chenCoeff.setGrids(self.localGrids)
    self._wfn.setGrids(self.localGrids[1:])

    # Tunnel matrix object
    self._tunnelMatrix = TunnelMatrix(self._chenCoeff, self._wfn)

    dimGrid = self.dimLocal
    # Currents for all bias voltages
    self.localCurrent = np.zeros(dimGrid+(len(self._voltages),), \
      dtype=ctypes.c_double)


    # Notes: The results can be made to match PPSTM code by:
    #        1. Remove the check for occupied / unoccupied.
    #        2. Add a check if eigTip in [0,voltage] (not sign of voltage).
    #        3. Remove the voltage from the DOS argument (i.e. only eigTip-eigSam).
    #       (4.) Remove additional check on DOS, PPSTM does not use cutoffs.

    # Over each seperate tunnel process (e.g. to O- or C-atom)
    for tunnelIdx in range(self._chenCoeff.noTunnels):
      for spinTipIdx in range(self._chenCoeff.noSpins):
        for etIdx in range(self._chenCoeff.noEigs[spinTipIdx]):
          eigTip = self._chenCoeff.eigs[spinTipIdx][etIdx]
          for spinSamIdx in range(self._wfn.noSpins):
            for esIdx in range(self._wfn.noEigs[spinSamIdx]):
              eigSample = self._wfn.eigs[spinSamIdx][esIdx]
              # Only if tip and sample are occupied/unoccupied
              if eigTip*eigSample > 0.0 \
                or (eigTip == 0.0 and eigSample <= 0.0) \
                or (eigSample == 0.0 and eigTip <= 0.0):
                continue

              # Checking if density of states will be 0 anyway
              skip = True
              for voltage in self._voltages:
                dosArgsSample = self._DosArgs(voltage+eigTip-eigSample)
                if self._dosType == "Gaussian" and \
                  abs(dosArgsSample.ene) < 4.0*dosArgsSample.sigma \
                  or self._dosType == "Lorentzian" and \
                  abs(dosArgsSample.ene) < 4.0*dosArgsSample.eta:
                  skip = False
                  break
              if skip:
                continue

              # Otherwise compute tunneling matrix entry
              tunnelMatrixSquared = (self._tunnelMatrix[tunnelIdx,spinTipIdx,etIdx, \
                spinSamIdx,esIdx])**2
              for volIdx, voltage in enumerate(self._voltages):
                dosArgsSample = self._DosArgs(voltage+eigTip-eigSample)
                if self._dosType == "Gaussian" and \
                  abs(dosArgsSample.ene) >= 4.0*dosArgsSample.sigma \
                  or self._dosType == "Lorentzian" and \
                  abs(dosArgsSample.ene) >= 4.0*dosArgsSample.eta:
                  continue
                #if voltage > 0 and (eigTip <= 0 or eigTip > voltage+0.001) \
                #or voltage <= 0 and (eigTip > 0 or eigTip < voltage-0.001):
                #  continue
                self.localCurrent[:,volIdx] += np.sign(eigTip)*tunnelMatrixSquared \
                  * self._dosSample(dosArgsSample)

  ##############################################################################
  def gather(self):
    """!
      @brief Gathers the current and returns it on rank 0.

      The current is a 4-dimensional array in the form [zIdx,yIdx,xIdx,vIdx].
    """
    if self.rank == 0:
      # Storage for current
      current = np.empty(self.dimGrid+(len(self._voltages),))
      # Relative positions of data (viewed as C-like array)
      outputLengths = self.splitLengths // 3 * len(self._voltages)
      offsetOutput = np.insert(np.cumsum(outputLengths),0,0)[:-1]
    else:
      current = None
      outputLengths = None
      offsetOutput = None
    # Broadcasting information on output
    outputLengths = self.comm.bcast(outputLengths, root=0)
    offsetOutput = self.comm.bcast(offsetOutput, root=0)
    # Gather currents
    self.comm.Gatherv(self.localCurrent, [current, outputLengths, offsetOutput, \
      MPI.DOUBLE], root=0)
    if self.rank == 0:
      return current
    else:
      return None

  def write(self, filename):
    """!
      @brief Writes the current to a file (*.npy).

      The file is written as a 1-dimensional array. The reconstruction has
      thus be done by hand. It can be reshaped into a 4-dimensional array in
      the form [zIdx,yIdx,xIdx,vIdx].

      @param filename Name of file.
    """
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    fh = MPI.File.Open(self.comm, filename, amode)
    fh.Write_at_all(self.offsets[self.rank], self.localCurrent)
    fh.Close()

  def write_compressed(self, filename, tol=1e-3):
    """!
      @brief Writes the current compressed to a file (*.npz).

      The file is written as a 1-dimensional array similar to write().
      Furthermore, in order to load the current use np.load()['arr_0'].

      @attention This method evokes a gather!

      @param filename Name of file.
      @param tol      Relative toleranz to the maximum for a height 
                      and voltage.
    """
    # Gather the current
    current = self.gather()

    if self.rank == 0:
      noVoltages = len(self._voltages)
      totDim = self.dimGrid+(noVoltages,)
      # Remove unnecessary data for better compression
      for hIdx in range(self.dimGrid[2]):
        for vIdx in range(noVoltages):
          maxVal = np.max(np.abs(np.reshape(current,totDim)[:,:,hIdx,vIdx]))
          # Note that this does not copy the data!
          tmp = np.reshape(current,totDim)[:,:,hIdx,vIdx]
          tmp[np.abs(tmp) < tol*maxVal] = 0.0
      # Save as 1-dimensional array to mimic write()
      np.savez_compressed(filename, current.ravel())
   

  ##############################################################################
  def run(self, voltages):
    """!
      @brief Performs the HR-STM simulation.

      @param voltages     List for bias voltages used in simulation in eV.
    """
    self._voltages = np.array(voltages, dtype=ctypes.c_double)

    # Broadcast output size here to avoid unnecessary barriers later
    if self.rank == 0:
      # Relative positions of data (viewed as C-like array)
      outputLengths = self.splitLengths // 3 * len(self._voltages) \
        * self._voltages.itemsize
      self.offsets= np.insert(np.cumsum(outputLengths),0,0)[:-1]
    else:
      self.offsets = None
    # Broadcasting information on output
    self.offsets = self.comm.bcast(self.offsets, root=0)
    
    self._compute()

################################################################################

################################################################################
################################################################################

################################################################################
class STM_CPP(STM):
  """!
    @brief Class that runs a STM simulation. Current is completely computed
           in C++.

    This class inherits from STM.
  """

  def _compute(self):
    """!
      @brief Evaluates the current for the processes. The current is stored
             in self.localCurrent and needs to be assembled.
    """
    array1d = np.ctypeslib.ndpointer(dtype=np.double,  ndim=1, flags='CONTIGUOUS')
    array1b = np.ctypeslib.ndpointer(dtype=np.bool_,  ndim=1, flags='CONTIGUOUS')

    path = os.path.dirname(os.path.abspath(__file__))
    clib = ctypes.CDLL(path+"/../cpp/libcurrent.so")
    # Define signature for C++ functions
    clib.computeCurrent.argtypes = [ \
      # wn, sigma, rot
      ctypes.c_double, ctypes.c_double, ctypes.c_bool, \
      #  rCut, pbc, abc
      ctypes.c_double, array1b, array1d, \
      # voltages, coeffsTip, coeffsSam
      ctypes.py_object, ctypes.py_object, ctypes.py_object, \
      # eigsTip, eigsSam,
      ctypes.py_object, ctypes.py_object, \
      # grids, atoms, current
      ctypes.py_object, ctypes.py_object, ctypes.py_object ]
    # Currents for all bias voltages
    self.localCurrent = np.zeros(self.dimLocal+(len(self._voltages),), \
      dtype=ctypes.c_double)

    if not self._voltages.flags["C_CONTIGUOUS"] \
      or not self._voltages.flags["ALIGNED"]:
      raise ValueError("Voltages are not C-contiguous or aligned.")
    for grid in self.localGrids:
      if not grid.flags["C_CONTIGUOUS"] or not grid.flags["ALIGNED"]:
        raise ValueError("Local grids are not C-contiguous or aligned.")
    if not self.localCurrent.flags["C_CONTIGUOUS"] \
      or not self.localCurrent.flags["ALIGNED"]:
      raise ValueError("Current storage failed to be C-contiguous or aligned.")

    self.sigma = self._DosArgs(0).sigma

    # Compute current
    clib.computeCurrent( self._wfn.wn, self.sigma, self._chenCoeff.rotate, \
      self._wfn.rcut, self._wfn.pbc, self._wfn.abc, \
      self._voltages, self._chenCoeff.singles, self._wfn.coefficients, \
      self._chenCoeff.eigs, self._wfn.eigs, \
      self.localGrids, self._wfn.atoms.get_positions(), self.localCurrent)

################################################################################
