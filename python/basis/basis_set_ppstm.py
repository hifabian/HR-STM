# @author Hillebrand, Fabian
# @date   2018-2019

from . import basis_set_abc

import os
import ctypes
import time
import numpy as np

eVToHartree = 1./27.21138602

def cp2kToPPSTM(basisCP2K, coeffsCP2K, atoms, noOrbs, upto=None):
  """!
    @brief Converts a CP2K basis coefficients into the PPSTM basis
           coefficients.

    @param basisCP2K  Dictionary for CP2K basis.
    @param coeffsCP2K Coefficients for the CP2K basis.
    @param atoms      Atom positions and symbols.
    @param noOrbs     Maximal orbital on sample ({0,1,...} for {s-,p-,...}).
    @param upto       Restricts converting coefficients for the first few atoms.

    @param Coefficients adjusted for the PPSTM basis.
  """

  noAtoms = len(coeffsCP2K)
  noSpins = len(coeffsCP2K[0])
  noEigs = [len(coeffsCP2K[0][spinIdx][0,:]) for spinIdx in range(noSpins)]
  noDer = (noOrbs+1)**2

  if upto is not None:
    noAtoms = upto

  # Build container
  coefficients = [None]*noSpins
  for spinIdx in range(noSpins):
    coefficients[spinIdx] = np.zeros((noAtoms,noEigs[spinIdx],noDer))

  # Loop over whole basis to determine which coefficients belong to which
  # orbitals
  for atomIdx in range(noAtoms):
    element = atoms[atomIdx].symbol
    coeffIdx = 0
    for basisIdx, basisSet in enumerate(basisCP2K[element]):
      noBasis = basisSet.noFunctions

      # Evaluate basis set to get coefficient
      for lIdx, l in enumerate(range(basisSet.lMin, basisSet.lMax+1)):
        for contrIdx in range(basisSet.contr[lIdx]):
          for mIdx in range(2*l+1):
            if l <= noOrbs:
              for spinIdx in range(noSpins):
                coefficients[spinIdx][atomIdx,:,l*l+mIdx] += \
                  coeffsCP2K[atomIdx][spinIdx][coeffIdx]
            coeffIdx += 1

  return coefficients

################################################################################
class BasisSetPPSTM(basis_set_abc.BasisSetAbstract):
  """!
    @brief Basis sets based on PPSTM basis. The exponent is given
           by the decay constant.

    The basis functions are based on the paper from Prokop Hapala et al:
    "Principles and simulations of high-resolution STM imaging with a 
     flexible tip apex" (2017).
  """

  def __init__(self, workfunction, noOrbs):
    """!
      @brief Constructor.

      @param workfunction Workfunction of the sample in eV.
      @param noOrbs       Maximal orbital on sample.
    """
    self.kappa = (2.*workfunction*eVToHartree)**0.5
    self._noOrbs = noOrbs

  ##############################################################################
  ## ABSTRACT METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ## Class methods
  @classmethod
  def from_file(cls, fnames):
    """! @attention Dummy function to intialize container. """
    workfunction = fnames[0]
    atoms = fnames[1]
    noOrbs = fnames[2]

    basisSets = {}
    for atomIdx in range(len(atoms)):
      element = atoms[atomIdx].symbol
      try:
        basisSets[element]
      except KeyError:
        basisSets[element] = [BasisSetPPSTM(workfunction, noOrbs)]

    return basisSets

  ##############################################################################
  ## Member variables
  @property
  def noFunctions(self):
    return (self._noOrbs+1)**2

  @property
  def lMin(self):
    return 0
    
  @property
  def lMax(self):
    return self._noOrbs

  ##############################################################################
  ## Member methods
  def evaluate(self, grid, distSquared, basisCoeffs, noOrbsTip):

    noSpins = len(basisCoeffs)
    noEigs = [np.shape(basisCoeffs[spinIdx])[0] for spinIdx in range(noSpins)]
    dimGrid = np.shape(grid)[:-1]
    noDer = (noOrbsTip+1)**2
    noOrbsSample = int( np.shape(basisCoeffs[0])[1]**0.5-1 )
    assert noOrbsSample == self._noOrbs, "Basis function coefficients does not" \
    + " match provided number of orbitals."
    # Distance
    dist = distSquared**0.5
    # Inverse distance
    distInv = 1. / dist
    # Prefactor
#    prefactor = 2.0*(np.pi*self.kappa)**0.5*np.exp(-self.kappa*dist) * distInv
    prefactor = np.exp(-self.kappa*dist) * distInv

    # Storage container
    wfnContribution = [np.zeros((noEigs[spinIdx],noDer)+dimGrid) \
      for spinIdx in range(noSpins)]

    for spinIdx in range(noSpins):
      ### s-orbital on tip
      tmp  = np.outer(basisCoeffs[spinIdx][:,0], dist)
      if noOrbsSample > 0: ### p-orbitals on sample
        tmp += \
            np.outer(3**0.5*basisCoeffs[spinIdx][:,1], grid[:,1]) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,2], grid[:,2]) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,3], grid[:,0])
      tmp = np.multiply(tmp.reshape((noEigs[spinIdx],)+dimGrid), prefactor)
      wfnContribution[spinIdx][:,0] += tmp

      ### p-orbitals on tip.
      if noOrbsTip > 0:
        ## py-orbital on tip
        tmp = np.outer(self.kappa*basisCoeffs[spinIdx][:,0], grid[:,1])
        if noOrbsSample > 0: ### p-orbitals on sample
          tmp += \
            np.outer(3**0.5*basisCoeffs[spinIdx][:,1], grid[:,1]**2*(self.kappa+distInv)*distInv-1) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,2], grid[:,2]*grid[:,1]*(self.kappa+distInv)*distInv) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,3], grid[:,0]*grid[:,1]*(self.kappa+distInv)*distInv)
        if noOrbsSample > 1: ### d-orbitals on sample
          pass
        tmp = np.multiply(tmp.reshape((noEigs[spinIdx],)+dimGrid), prefactor)
        wfnContribution[spinIdx][:,1] += tmp
        ## pz-orbital on tip
        tmp = np.outer(self.kappa*basisCoeffs[spinIdx][:,0], grid[:,2])
        if noOrbsSample > 0: ### p-orbitals on sample
          tmp += \
            np.outer(3**0.5*basisCoeffs[spinIdx][:,1], grid[:,1]*grid[:,2]*(self.kappa+distInv)*distInv) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,2], grid[:,2]**2*(self.kappa+distInv)*distInv-1) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,3], grid[:,0]*grid[:,2]*(self.kappa+distInv)*distInv)
        if noOrbsSample > 1: ### d-orbitals on sample
          pass
        tmp = np.multiply(tmp.reshape((noEigs[spinIdx],)+dimGrid), prefactor)
        wfnContribution[spinIdx][:,2] += tmp
        ## px-orbital on tip
        tmp = np.outer(self.kappa*basisCoeffs[spinIdx][:,0], grid[:,0])
        if noOrbsSample > 0: ### p-orbitals on sample
          tmp += \
            np.outer(3**0.5*basisCoeffs[spinIdx][:,1], grid[:,1]*grid[:,0]*(self.kappa+distInv)*distInv) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,2], grid[:,2]*grid[:,0]*(self.kappa+distInv)*distInv) \
          + np.outer(3**0.5*basisCoeffs[spinIdx][:,3], grid[:,0]**2*(self.kappa+distInv)*distInv-1) 
        if noOrbsSample > 1: ### d-orbitals on sample
          pass
        tmp = np.multiply(tmp.reshape((noEigs[spinIdx],)+dimGrid), prefactor)
        wfnContribution[spinIdx][:,3] += tmp

      ### d-orbitals on tip
      if noOrbsTip > 1:
        raise NotImplementedError("d-orbitals on tip not supported.")
        
    return wfnContribution

################################################################################
