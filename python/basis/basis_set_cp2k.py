# @author Hillebrand, Fabian
# @date   2018-2019

from . import basis_set_abc
from maths.spherical_harmonics import *

import numpy as np

################################################################################
class BasisSetCP2K(basis_set_abc.BasisSetAbstract):
  """!
    @brief Basis set compatible with CP2K.

    The basis set is built up by contracted Gaussians and real spherical 
    harmonics.
  """

  def __init__(self, noBasis, lMin, lMax, contr, exps, prefs):
    """!
      @brief Constructor.

      @param noBasis Total number of basis functions in this basis set.
      @param lMin    Minimal l (relating to spherical harmonics).
      @param lMax    Maximal l (relating to spherical harmonics).
      @param contr   An array determining the number of contracted Gaussians
                     belonging to same l-value.
      @param exps    An array containing the exponents.
      @param prefs   A matrix containing the prefactors.
    """
    self.noBasis = noBasis
    self._lMin = lMin
    self._lMax = lMax
    self.contr = contr
    self.exps = exps
    self.prefs = prefs


  ##############################################################################
  ## Class methods
  @classmethod
  def _basis_renorm(cls, basisSets):
    """!
      @brief Normalizes the basis sets as read from CP2K.

      @param basisSets A dictionary of all basis sets for each element.

      @return Basis sets container similar to input but coefficients are
              normalized.

      @note Taken from Kristjan Eimre and adjusted for needs.
            See: https://github.com/eimrek/atomistic_tools
    """
    for elem in basisSets:
      for bset in basisSets[elem]:
        noExps = len(bset.exps)
        basisIdx = 0
        # Over all l
        for lIdx, l in enumerate(range(bset.lMin, bset.lMax+1)):
          # All contracted with this l
          for contrIdx in range(bset.contr[lIdx]):
            normFactor = 0
            for i in range(noExps-1):
              for j in range(i+1, noExps):
                normFactor += 2*bset.prefs[i,basisIdx]*bset.prefs[j,basisIdx] \
                  * ( 2*np.sqrt(bset.exps[i]*bset.exps[j]) \
                  / (bset.exps[i]+bset.exps[j]) )**((2*l+3)/2)
            for i in range(noExps):
              normFactor += bset.prefs[i, basisIdx]**2
            for i in range(noExps):
              bset.prefs[i, basisIdx] = bset.prefs[i,basisIdx] \
                * bset.exps[i]**((2*l+3)/4)/np.sqrt(normFactor)
            basisIdx += 1
      
    return basisSets
    
  @classmethod
  def _read_cp2k_input(cls, fname):
    """! @brief Reads a CP2K input file to extract basis names. """
    e2bn = {}

    with open(fname) as f:
      lines = f.readlines()
      for lineIdx in range(len(lines)):
        parts = lines[lineIdx].split()
        if len(parts) == 0:     # Skipping empty lines
          continue
        if parts[0] == "&KIND": # Found a basis set
          element = parts[1]
          j = 1
          while True: # Search until basis name found (for comments/potentials)
            parts = lines[lineIdx+j].split()
            if parts[0] == "BASIS_SET":
              e2bn[element] = parts[1]
              break
            j += 1

    return e2bn

  @classmethod
  def _read_basis_sets(cls, fname, e2bn):
    """! @brief Reads a CP2K basis set file and extracts all basis sets. """
    basisSets = {}

    with open(fname) as f:
      lines = f.readlines()
      for lineIdx in range(len(lines)):
        parts = lines[lineIdx].split()
        if len(parts) == 0:  # Skipping empty lines
          continue
        if parts[0] in e2bn: # Found a basis set
          element = parts[0]
          # Found the basis set for element
          if parts[1] == e2bn[element] \
            or (len(parts) > 2 and parts[2] == e2bn[element]):
            noSets = int(lines[lineIdx+1])
            elementSets = [] # List for basis sets on this element

            offset = 2 # Offset to line index
            for setIdx in range(noSets):
              composition = [int(x) for x in lines[lineIdx+offset].split()]
              _, lMin, lMax, numExp = composition[:4]
              contr = composition[4:]
              offset += 1

              noBasis = 0
              for lIdx, l in enumerate(range(lMin, lMax+1)):
                noBasis += contr[lIdx]*(2*l+1)

              exps = np.empty(numExp)
              for idx in range(numExp):
                exps[idx] = float(lines[lineIdx+offset+idx].split()[0])

              prefs = np.empty([numExp, sum(composition[4:])])
              for idx in range(numExp):
                coeffs = [float(x) for x in lines[lineIdx+offset+idx].split()]
                for jdx in range(sum(composition[4:])):
                  prefs[idx, jdx] = coeffs[jdx+1]

              offset += numExp

              elementSets.append(BasisSetCP2K(noBasis,lMin,lMax,contr,exps,prefs))

            # Insert basis sets on element into dictionary of all basis sets
            basisSets[element] = elementSets

    return cls._basis_renorm(basisSets)


  ##############################################################################
  ## Member methods
  def __repr__(self):
    return self.__str__()
  def __str__(self):
    return "["+str(self._lMin)+", " \
              +str(self._lMax)+", " \
              +str(self.contr)+", " \
              +str(self.exps)+", " \
              +str(self.prefs)+"]"


  ##############################################################################
  ## ABSTRACT METHODS AND VARIABLES
  ##############################################################################

  ##############################################################################
  ## Class methods
  @classmethod
  def from_file(cls, fnames):
    """!
      @param fnames A list containing the CP2K input filename and the basis set
                    input filename (which contains the appropriate basis sets).
    """
    fname_cp2k_input = fnames[0]
    fname_basis_sets = fnames[1]

    # Extract element to basis name dictionary from CP2K input
    e2bn = cls._read_cp2k_input(fname_cp2k_input)
    # Create dictionary for all basis sets from basis set input
    allBasisSets = cls._read_basis_sets(fname_basis_sets, e2bn)

    return allBasisSets


  ##############################################################################
  ## Member variables
  @property
  def noFunctions(self):
    return self.noBasis
  @property
  def lMin(self):
    return self._lMin
  @property
  def lMax(self):
    return self._lMax


  ##############################################################################
  ## Member methods
  def evaluate(self, grid, distSquared, basisCoeffs, orbsTip):
    raise NotImplementedError("This is not compatible with WavefunctionLCAO.")
    # As of now, this is incompatible with WavefunctionLCAO due to the
    # structure of the basisCoeffs argument (this cannot be fixed without
    # braking everything else.)
    """
    # Gaussian at grid points for all exponents
    contrExp = np.exp(np.einsum("ijk,l->ijkl", distSquared,-self.exps))

    noSpins = len(basisCoeffs)
    noEigs = [len(basisCoeffs[spinIdx][0]) for spinIdx in range(noSpins)]
    dimGrid = np.shape(grid)[:-1]
    derSize = (orbsTip+1)**2

    wfnContribution = [np.zeros((derSize,noEigs[spinIdx],)+dimGrid) \
      for spinIdx in range(noSpins)]

    # Total index for contracted Gaussian
    gaussIdx = 0
    coeffIdx = 0
    for lIdx, l in enumerate(range(self.lMin, self.lMax+1)):

      # Spherical component
      sphericalComp = np.empty((2*l+1,)+dimGrid)
      if orbsTip > 0:
        sphericalCompDx = np.empty((2*l+1,)+dimGrid)
        sphericalCompDy = np.empty((2*l+1,)+dimGrid)
        sphericalCompDz = np.empty((2*l+1,)+dimGrid)
      for mIdx, m in enumerate(range(-l, l+1)):
        sphericalComp[mIdx] = spherical_harmonic(l,m,grid)
        if orbsTip > 0:
          sphericalCompDx[mIdx] = spherical_harmonic_d0(l,m,grid)
          sphericalCompDy[mIdx] = spherical_harmonic_d1(l,m,grid)
          sphericalCompDz[mIdx] = spherical_harmonic_d2(l,m,grid)

      for contrIdx in range(self.contr[lIdx]):
        # Radial part
        radialComp = np.einsum("k,mnlk->mnl",self.prefs[:,gaussIdx],contrExp)
        if orbsTip > 0:
          # Derivative or radial part without grid dependency
          radialCompD = np.einsum("k,mnlk->mnl",-2*self.exps[gaussIdx] \
            * self.prefs[:,gaussIdx],contrExp)
        gaussIdx += 1

        # Loop over all m for these orbitals
        for mIdx in range(2*l+1):
          basisComp = sphericalComp[mIdx]*radialComp
          if orbsTip > 0:
            basisCompDx = sphericalCompDx[mIdx]*radialComp \
              + radialCompD*grid[0]
            basisCompDy = sphericalCompDy[mIdx]*radialComp \
              + radialCompD*grid[1]
            basisCompDz = sphericalCompDz[mIdx]*radialComp \
              + radialCompD*grid[2]

          for spinIdx in range(noSpins):
            wfnContribution[spinIdx][0] += np.einsum("i,jkl->ijkl", \
              basisCoeffs[spinIdx][coeffIdx],basisComp)

            if orbsTip > 0:
              wfnContribution[spinIdx][1] += np.einsum("i,jkl->ijkl", \
                  basisCoeffs[spinIdx][coeffIdx],basisCompDy)
              wfnContribution[spinIdx][2] += np.einsum("i,jkl->ijkl", \
                  basisCoeffs[spinIdx][coeffIdx],basisCompDz)
              wfnContribution[spinIdx][3] += np.einsum("i,jkl->ijkl", \
                  basisCoeffs[spinIdx][coeffIdx],basisCompDx)
          coeffIdx += 1

    return wfnContribution
    """
    pass

################################################################################
