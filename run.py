#!/usr/bin/env python

# @author Hillebrand, Fabian
# @date   2018-2019

# Purpose: Run the HR-STM code.

#        ___          ___                  ___       ___          ___
#       /  /\        /  /\                /\  \     /\  \        /\__\
#      /  / /       /  /  \              /  \  \    \ \  \      /  |  |
#     /  / /       /  / /\ \            / /\ \  \    \ \  \    / | |  |
#    /  /  \ ___  /  /  \ \ \          _\ \-\ \  \   /  \  \  / /| |__|__
#   /__/ /\ \  /\/__/ /\ \_\ \  ____  /\ \ \ \ \__\ / /\ \__\/ / |    \__\
#   \__\/  \ \/ /\__\/-|  \/ / |    | \ \ \ \ \/__// /  \/__/\/__/--/ /  /
#        \__\  /    |  | |  /  |____|  \ \ \ \__\ / /  /           / /  /
#        /  / /     |  | |\/            \ \/ /  / \/__/           / /  /
#       /__/ /      |__| |               \  /  /                 / /  / 
#       \__\/        \__\|                \/__/                  \/__/
#
                                          
# ==============================================================================
# ------------------------------------------------------------------------------
#                                   INCLUDES
# ------------------------------------------------------------------------------
# ==============================================================================

import argparse # Terminal arguments
import sys      # System access
import time     # Timing
import resource # Memory usage

import os

# Python3 at least
assert sys.version_info >= (3, 0)

import numpy as np
import sysconfig as sc

# Include directory
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/python/")

# Own includes
from util import *                        # Various methods that have no home
from stm import *                     # STM classes
from read_input import *                  # Read input files
from basis.basis_set_cp2k import *        # CP2K Basis sets
from basis.basis_set_ppstm import *       # PPSTM Basis sets
from basis.wavefunction_cpp import *      # C++ implemented wavefunction
from tunnelling.chen_coeffs_cpp import *  # C++ implemented derivative rule

from mpi4py import MPI # MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

startTotal = time.time()


# The proper way is to use -m mpi4py but this failed on the cluster
def global_excepthook(etype, value, tb):
  try:
    sys.stderr.write("Uncaught exception on rank {}.\n".format(rank))
    from traceback import print_exception
    print_exception(etype, value, tb)
    sys.stderr.write("Aborting program.\n")
    sys.stderr.flush()
  finally:
    try:
      comm.Abort(1)
    except Exception as e:
      sys.stderr.write("Failed to abort MPI processes! Please kill manually.\n")
      sys.stderr.flush()
      raise e
sys.excepthook = global_excepthook

# ==============================================================================
# ------------------------------------------------------------------------------
#                              READ ARGUMENT LINE
# ------------------------------------------------------------------------------
# ==============================================================================

parser = argparse.ArgumentParser(description="HR-STM for CP2K based on Chen's"
  + " Derivative Rule and the Probe Particle Model.")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

parser.add_argument('--output',
  type=str,
  metavar="OUTPUT",
  required=True,
  help="Name for output file stored as 'OUTPUT.npy'.")

# ------------------------------------------------------------------------------

parser.add_argument('--cp2k_input_s',
  metavar='FILE',
  required=True,
  help="CP2K input file used for the sample calculation.")

parser.add_argument('--basis_sets_s',
  metavar='FILE',
  required=True,
  help="File containing all basis sets used for sample.")

parser.add_argument('--xyz_s',
  metavar='FILE',
  required=True,
  help="File containing the atom positions for the sample.")

parser.add_argument('--coeffs_s',
  metavar='FILE',
  required=True,
  help="File containing the basis coefficients for the sample"
  + " (*.wfn or *.MOLog).")

parser.add_argument('--orbs_sam',
  type=int,
  default=1,
  metavar="l",
  required=False,
  help="Integer indicating which orbitals of the sample wavefunction are used"
  + " following the angular momentum quantum number used by PPSTM basis.")

parser.add_argument('--pbc',
  type=int,
  metavar='N',
  nargs=3,
  default=[0,0,0],
  required=False,
  help="Integers indicating if periodic boundaries are to be applied.")

parser.add_argument('--rcut',
  type=float,
  metavar='A',
  default=15.0,
  required=False,
  help="Cutoff radius used when computing sample wavefunction.")

# ------------------------------------------------------------------------------

parser.add_argument('--tip_pos',
  metavar='FILE',
  nargs='+',
  required=False,
  help="Paths to files containing relaxed positions of tip apexes."
  + " Files are assumed to end with 'FILE_*.npy'."
  + " Reference grids are taken as the proceeding grids with the"
  + " first using a uniform grid which does not need to be given.")

parser.add_argument('--tip_shift',
  type=float,
  metavar='A',
  nargs="+",
  required=True,
  help="z-distance shift for the metal tip with respect to apex"
  + " in Angstrom.")

parser.add_argument('--heights',
  type=float,
  default=[],
  nargs='+',
  metavar='A',
  required=False,
  help="Constant heights for scan in Angstrom. If set, grid is restricted"
  + " to the closest heights available otherwise the complete grid is evaluated.")

parser.add_argument('--eval_region',
  type=float,
  metavar='A',
  nargs=6,
  required=False,
  help="Evaluation region in Angstrom used when using non-relaxed grid:"
  + " [xMin xMax yMin yMax zMin zMax]")

parser.add_argument('--eval_dim',
  type=int,
  metavar='N',
  nargs=3,
  required=False,
  help="Dimensions of grid if using non-relaxed grid.")

# ------------------------------------------------------------------------------

parser.add_argument('--orbs_tip',
  type=int,
  metavar="l",
  required=True,
  help="Integer indicating which orbitals of the tip are used following the"
  + " angular momentum quantum number.")

parser.add_argument('--rotate',
  action="store_true", 
  default=False,
  required=False,
  help="If set, tip coefficients will be rotated.")

parser.add_argument('--pdos_list',
  metavar='FILE',
  nargs='+',
  required=False,
  help="List of PDOS files for the different tip apexes used as tip"
  + " coefficients. Or, alternatively, five numbers corresponding to"
  + " [s py pz px de] for uniform PDOS values whose energies are spaced"
  + " de apart from each other." )

parser.add_argument('--cp2k_input_t',
  metavar='FILE',
  required=False,
  help="CP2K input file used for the tip calculation."
  + " Alternative to using PDOS.")

parser.add_argument('--basis_sets_t',
  metavar='FILE',
  required=False,
  help="File containing all basis sets used for tip."
  + " Alternative to using PDOS.")

parser.add_argument('--xyz_t',
  metavar='FILE',
  required=False,
  help="File containing atom positions for the tip."
  + " Alternative to using PDOS.")

parser.add_argument('--coeffs_t',
  metavar='FILE',
  required=False,
  help="File containing the basis coefficients for the tip (*.wfn or *.MOLog)."
  + " Alternative to using PDOS.")

parser.add_argument('--tip_ids',
  type=int,
  nargs='+',
  required=False,
  help="Indices for tip apex atoms used for tunnelling."
  + " Alternative to using PDOS.")

# ------------------------------------------------------------------------------

parser.add_argument('--voltages',
  type=float,
  metavar='U',
  nargs='+',
  required=True,
  help="Voltages used for STM in eV.")

# ------------------------------------------------------------------------------

parser.add_argument('--emin',
  type=float,
  metavar='E',
  required=False,
  help="Lower energy used for cutoff with respect to Fermi energy in eV"
  + " for sample.")

parser.add_argument('--emax',
  type=float,
  metavar='E',
  required=False,
  help="Upper energy used for cutoff with respect to Fermi energy in eV"
  + " for sample.")

parser.add_argument('--etip',
  type=float,
  default=0.0,
  metavar='E',
  required=False,
  help="Energy range added to the maximal and minimal bias voltages for"
  + " the tip energies in eV.")

parser.add_argument('--workfunction',
  type=float,
  default=5.0,
  metavar='U',
  required=False,
  help="Value for workfunction in eV.")

parser.add_argument('--dos',
  default='Gaussian',
  metavar='STRING',
  required=False,
  help="Specifies type of density of states used in place of dirac-delta"
  + " functions. Currently only Gaussians are supported by C++ code.")

parser.add_argument('--eta',
  type=float,
  default=0.05,
  metavar='E',
  required=False,
  help="Half of the broading factor used for Lorentzian density of states.")

parser.add_argument('--fwhm',
  type=float,
  default=0.1,
  metavar='E',
  required=False,
  help="Full width at half maximum used for Gaussian density of states.")

# ------------------------------------------------------------------------------

parser.add_argument('--scaling',
  type=float,
  default=1.0,
  required=False,
  help="Scaling of the molecule coefficients.")

# TODO if this is to be formalised, this should be done automatically ideally
parser.add_argument('--mol',
  type=int,
  default=0,
  required=False,
  help="Index of last atom belonging to molecule (assumed at the start)")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ==============================================================================
# ------------------------------------------------------------------------------
#              READ FILES AND SET UP INPUT WITH PROCESS RANK 0 ONLY
# ------------------------------------------------------------------------------
# ==============================================================================
if rank == 0:

  args = parser.parse_args()

  # Check if necessary information was provided

  # No scanning grid
  if args.tip_pos is None:
    if args.eval_region is None \
      or args.eval_dim is None:
      raise KeyError("Missing tip positions. Please provide either --tip_pos"
        + " or --eval_region and --eval_dim.")
    if args.pdos_list is None and len(args.tip_ids) > len(args.tip_shift):
      raise KeyError("More atoms requested than positions provided. Please"
        + " provide more arguments for --tip_shift.")
  # Number of arguments for grid do not match
  elif args.pdos_list is None and len(args.tip_ids) > len(args.tip_pos):
      raise KeyError("More atoms requested than positions provided. Please"
        + " provide more arguments for --tip_pos.")
  # Missing Chen's coefficients
  if args.pdos_list is None:
    if args.cp2k_input_t is None \
      or args.basis_sets_t is None \
      or args.xyz_t is None \
      or args.coeffs_t is None \
      or args.tip_ids is None:
      raise KeyError("Missing tip coefficients. Please provide either"
        + " --pdos_list or --cp2k_input_s and --basis_sets_t and --xyz_t"
        + " and --coeffs_t and --tip_ids.")

  # ============================================================================
  # ----------------------------------------------------------------------------
  #                                LOAD INPUT FILES
  # ----------------------------------------------------------------------------
  # ============================================================================

  # ----------------------------------------------------------------------------
  # READING SAMPLE INFORMATION
  # ----------------------------------------------------------------------------

  start = time.time()
  # ABC of sample
  abc = read_ABC(args.cp2k_input_s)
  # Coefficients
  if args.coeffs_s.endswith('.MOLog'):
    coeffsSam, eigsSam, _ = read_MO(args.coeffs_s, args.emin, args.emax)
  else:
    coeffsSam, eigsSam, _ = read_wfn(args.coeffs_s, args.emin, args.emax)
  # Basis sets (CP2K)
  elemToBasisSam = BasisSetCP2K.from_file([args.cp2k_input_s, args.basis_sets_s])
  # Positionsfs_s, args.emi
  atomsSam = read_xyz(args.xyz_s)
  # Basis sets (PPSTM set from CP2K)
  # TODO scale molecule coefficients by factor lamdba in [0,1]
  coeffsSam = cp2kToPPSTM(elemToBasisSam, coeffsSam, atomsSam, args.orbs_sam)
  coeffsSam = scale(coeffSam, args.scaling, args.mol)
  elemToBasisSam = BasisSetPPSTM.from_file([args.workfunction, atomsSam, \
    args.orbs_sam])
  end = time.time()
  print("Reading sample information in {} seconds.".format(end-start))

  # ----------------------------------------------------------------------------
  # READING EVALUATION GRID (SCAN)
  # ----------------------------------------------------------------------------

  tipPos = []

  start = time.time()
  # Relaxed grid
  if args.tip_pos is not None:
    for filename in args.tip_pos:
      tmp, lVec = read_PPPos(filename)
      tipPos.append(tmp)
    dim = np.shape(tipPos[0])[:-1]
    # Reference grid for first grid
    tipPos.insert(0, np.transpose( np.mgrid[ \
      lVec[0,0]:lVec[0,0]+lVec[1,0]:dim[0]*1j, \
      lVec[0,1]:lVec[0,1]+lVec[2,1]:dim[1]*1j, \
      lVec[0,2]-args.tip_shift[0]:lVec[0,2]+lVec[3,2]-args.tip_shift[0] \
      :dim[2]*1j], axes=(1,2,3,0) ).copy())
  # Non-relaxed grid
  else:
    dim = args.eval_dim
    for ids, shift in enumerate(args.tip_shift):
      tipPos.append(np.transpose( np.mgrid[ \
        args.eval_region[0]:args.eval_region[1]:args.eval_dim[0]*1j, \
        args.eval_region[2]:args.eval_region[3]:args.eval_dim[1]*1j, \
        args.eval_region[4]+shift:args.eval_region[5]+shift:args.eval_dim[2]*1j], \
        axes=(1,2,3,0) ).copy())
    # Last grid for final tip apex
    tipPos.append(np.transpose( np.mgrid[ \
        args.eval_region[0]:args.eval_region[1]:args.eval_dim[0]*1j, \
        args.eval_region[2]:args.eval_region[3]:args.eval_dim[1]*1j, \
        args.eval_region[4]:args.eval_region[5]:args.eval_dim[2]*1j], \
        axes=(1,2,3,0) ).copy())
    lVec = np.array([
      [args.eval_region[0],args.eval_region[2],args.eval_region[4]], 
      [args.eval_region[1]-args.eval_region[0],0.0, 0.0], 
      [0.0, args.eval_region[3]-args.eval_region[2], 0.0],
      [0.0, 0.0, args.eval_region[5]-args.eval_region[4]]])
  # Restrict grid to heights, if given
  trueHeights = []
  if args.heights:
    trueHeights, heightIds = getHeightIndices(args.heights, lVec, dim, atomsSam)
    for ids in range(len(tipPos)):
      tipPos[ids] = np.array(tipPos[ids][:,:,heightIds].copy())
  end = time.time()
  print("Reading grid in {} seconds.".format(end-start))

  # ----------------------------------------------------------------------------
  # READING CHEN'S COEFFICIENTS
  # ----------------------------------------------------------------------------

  # Energy limits for tip
  minETip = min(args.voltages)-args.etip
  maxETip = max(args.voltages)+args.etip

  chenSingles = []

  start = time.time()
  # Using PDOS
  if args.pdos_list is not None:
    idx = 0
    while idx < len(args.pdos_list):
      try:
        chenSingle, eigsTip = constCoeffs(minETip, maxETip, 
          s=float(args.pdos_list[idx]), \
          py=float(args.pdos_list[idx+1]), \
          pz=float(args.pdos_list[idx+2]), \
          px=float(args.pdos_list[idx+3]), \
          de=float(args.pdos_list[idx+4]))
        idx += 5
      except ValueError:
        chenSingle, eigsTip = read_PDOS(args.pdos_list[idx], minETip, maxETip)
        # Take square root to obtain proper coefficients
        for spinIdx in range(len(chenSingle)):
          chenSingle[spinIdx] = chenSingle[spinIdx][:,:(args.orbs_tip+1)**2]**0.5
        idx += 1
      chenSingles.append(chenSingle)
  # Using PPSTM coefficients (summed from CP2K)
  else:
    # Coefficients
    if args.coeffs_s.endswith('.MOLog'):
      coeffsTip, eigsTip, _ = read_MO(args.coeffs_t, minETip, maxETip)
    else:
      coeffsTip, eigsTip, _ = read_wfn(args.coeffs_t, minETip, maxETip)
    # Basis sets (CP2K)
    elemToBasisTip = BasisSetCP2K.from_file([args.cp2k_input_t, args.basis_sets_t])
    # Positions
    atomsTip = read_xyz(args.xyz_t)
    # Basis sets (PPSTM set from CP2K)
    coeffsTip = cp2kToPPSTM(elemToBasisTip, coeffsTip, atomsTip, args.orbs_tip, 2)
    del elemToBasisTip, atomsTip
    # Take basis coefficients as Chen's coefficients
    noSpins = len(coeffsTip)
    for idx in args.tip_ids:
      chenSingle = [None]*noSpins
      for spinIdx in range(noSpins):
        chenSingle[spinIdx] = coeffsTip[spinIdx][idx,:]
      chenSingles.append(chenSingle)
  end = time.time()
  print("Reading of Chen's coefficients in {} seconds.".format(end-start))

  # ----------------------------------------------------------------------------
  # CHEN'S COEFFICIENTS AND WAVEFUNCTION OBJECTS
  # ----------------------------------------------------------------------------

  # Compile C++ codes
  start = time.time()
  includes = "-I"+sc.get_paths()["include"] + " -I"+np.get_include()
  os.system("make --directory "+ os.path.dirname(os.path.realpath(__file__)) \
    +"/cpp INCLUDES='"+includes+"'")
  end = time.time()
  print("Compiling in {} seconds.".format(end-start))

  # Get Chen's coefficients on grid
  chenCoeffs = ChenCoeffsCPP(noOrbs=args.orbs_tip, singles=chenSingles, \
    eigs=eigsTip, rotate=args.rotate)
  # Get sample wavefunction
  wfnSam = WavefunctionCPP(workfunction=args.workfunction, rcut=args.rcut, \
    pbc=args.pbc, abc=abc, \
    noOrbsTip=args.orbs_tip, eigs=eigsSam, coefficients=coeffsSam, atoms=atomsSam)

  # ----------------------------------------------------------------------------
  # WRITING ADDITIONAL INFROMATION
  # ----------------------------------------------------------------------------

  # Meta information
  meta = {'dimGrid' : np.shape(tipPos[0])[:-1], \
          'lVec' : lVec, \
          'voltages' : args.voltages, \
          'heights' : np.array(args.heights), \
          'trueHeights' : np.array(trueHeights), \
  }
  np.save(args.output+"_meta.npy", meta)


# Remaining processes
else:
  chenCoeffs = None
  wfnSam = None
  atomsSam = None
  tipPos = None
  lVec = None
  args = None

# ==============================================================================
# ------------------------------------------------------------------------------
#                          BROADCASTING NECESSARY OBJECTS
# ------------------------------------------------------------------------------
# ==============================================================================

# Broadcast to other process
start = time.time()
chenCoeffs = comm.bcast(chenCoeffs, root=0)
wfnSam = comm.bcast(wfnSam, root=0)
atomsSam = comm.bcast(atomsSam, root=0)
lVec = comm.bcast(lVec, root=0)
args = comm.bcast(args, root=0)
end = time.time()
print("Broadcasting in {} seconds for rank {}.".format(end-start, rank))

# ==============================================================================
# ------------------------------------------------------------------------------
#                               CREATE STM OBJECT
# ------------------------------------------------------------------------------
# ==============================================================================

start = time.time()
if args.dos == 'Gaussian':
  stm = STM_CPP(chenCoeffs, wfnSam, atomsSam, tipPos, lVec, args.dos, \
    [args.fwhm], comm, size, rank)
  #stm = STM(chenCoeffs, wfnSam, atomsSam, tipPos, lVec, args.dos, \
  #  [args.fwhm], comm, size, rank)
elif args.dos == 'Lorentzian':
  stm = STM(chenCoeffs, wfnSam, atomsSam, tipPos, lVec, args.dos, \
    [args.eta], comm, size, rank)
else:
  raise KeyError("Could not understand density of state type of "+args.dos+".")
del tipPos # Saving memory primary for rank 0
stm.run(args.voltages)
end = time.time()
print("Evaluating STM-run method in {} seconds.".format(end-start))

#stm.write(args.output+".npy")
stm.write_compressed(args.output)

print("Maximum memory usage was {} kilobytes.".format( \
  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
endTotal = time.time()
print("Total time was {} seconds for rank {}.".format(endTotal-startTotal, \
  rank))

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================
