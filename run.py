#!/usr/bin/env python

# TODO: Plan of action:
#   Use Kristjan's wavefunction. For the derivative, derive the
#   interpolation if possible.

# @author  Hillebrand, Fabian
# @date    2019
# @version dev2.0.0

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
import atomistic_tools.cp2k_grid_orbitals as cgo
import atomistic_tools.cp2k_stm_sts as css
from atomistic_tools import common
from atomistic_tools.cube import Cube

from python.read_input import *
from python.util import *
from python.basis.wavefunction_cp2k import *
from python.tunnelling.chen_coeffs_python import *
from python.hrstm import *

# ------------------------------------------------------------------------------

ang2bohr   = 1.88972612546
ev2hartree = 0.03674930814

# ------------------------------------------------------------------------------

from mpi4py import MPI # MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

startTotal = time.time()

# ------------------------------------------------------------------------------

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

parser.add_argument('--output',
  type=str,
  metavar="OUTPUT",
  default="hrstm",
  required=False,
  help="Name for output file. File extension added automatically.")

# ------------------------------------------------------------------------------

parser.add_argument('--cp2k_input',
  metavar='FILE',
  required=True,
  help="CP2K input file used for the sample calculation.")

parser.add_argument('--basis_sets',
  metavar='FILE',
  required=True,
  help="File containing all basis sets used for sample calculation.")

parser.add_argument('--xyz',
  metavar='FILE',
  required=True,
  help="File containing the atom positions for the sample.")

parser.add_argument('--coeffs',
  metavar='FILE',
  required=True,
  help="Restart file containing the basis coefficients for the sample.")

parser.add_argument('--rcut',
  type=float,
  metavar='Ang',
  default=14.0,
  required=False,
  help="Cut-off radius used when computing sample wavefunction.")

parser.add_argument('--hartree_file',
  metavar='FILE',
  required=True,
  help="Cube file with Hartree potential of sample.")

# ------------------------------------------------------------------------------

parser.add_argument('--tip_pos',
  metavar='FILE',
  nargs='+',
  required=True,
  help="File paths to positions of probe particles.")

parser.add_argument('--tip_shift',
  type=float,
  metavar='Ang',
  required=True,
  help="z-distance shift for metal tip with respect to apex probe particle.")

# ------------------------------------------------------------------------------

parser.add_argument('--pdos_list',
  metavar='FILE',
  nargs='+',
  required=True,
  help="List of PDOS files for the different tip apexes used as tip"
  + " coefficients. Or, alternatively, five numbers corresponding to"
  + " [s py pz px de] for uniform PDOS values whose energies are spaced"
  + " de apart from each other." )

parser.add_argument('--orbs_tip',
  type=int,
  metavar="l",
  default=1,
  required=False,
  help="Integer indicating which orbitals of the tip are used following the"
  + " angular momentum quantum number.")

# ------------------------------------------------------------------------------

parser.add_argument('--voltages',
  type=float,
  metavar='eV',
  nargs='+',
  required=True,
  help="Voltages used for STM.")

# ------------------------------------------------------------------------------

parser.add_argument('--emin',
  type=float,
  metavar='eV',
  default=-2.0,
  required=False,
  help="Lower energy used for cut-off with respect to Fermi energy for sample.")

parser.add_argument('--emax',
  type=float,
  metavar='eV',
  default=2.0,
  required=False,
  help="Upper energy used for cut-off with respect to Fermi energy for sample.")

# TODO support this!
parser.add_argument('--dx',
  type=float,
  metavar='Ang',
  default=0.1,
  required=False,
  help="Spacing for grid used by interpolation.")

parser.add_argument('--extrap_extent',
  type=float,
  metavar='Ang',
  default=0.5,
  required=False,
  help="Extent of the extrapolation region.")

parser.add_argument('--fwhm',
  type=float,
  metavar='eV',
  default=0.05,
  required=False,
  help="Full width at half maximum for Gaussian broadening of sample states.")

# ==============================================================================
# ------------------------------------------------------------------------------
#                         READ FILES AND SET UP INPUT
# ------------------------------------------------------------------------------
# ==============================================================================

# TODO remove this
time0 = time.time()

# ------------------------------------------------------------------------------
# BROADCAST INPUT ARGUMENTS
# ------------------------------------------------------------------------------

args = None
if rank == 0:
  args = parser.parse_args()
args = comm.bcast(args, root=0)

# ------------------------------------------------------------------------------
# READ RELAXED GRID
# ------------------------------------------------------------------------------

# TODO Parallelization! Divide tip positions equally along x-direction to the
#      processes.

tipPos = []

start = time.time()
for filename in args.tip_pos:
  tmp, lVec = read_PPPos(filename)
  # Restrict tip positions to cell box and store it
  tipPos.append(apply_bounds(tmp,lVec))
dim = np.shape(tipPos[0])[1:]
# Metal tip (needed for rotation, no tunnelling considered)
# NOTE Not needed after determining the rotational matrices!
tipPos.insert(0, np.mgrid[ \
  lVec[0,0]:lVec[0,0]+lVec[1,0]:dim[0]*1j, \
  lVec[0,1]:lVec[0,1]+lVec[2,1]:dim[1]*1j, \
  lVec[0,2]-args.tip_shift:lVec[0,2]+lVec[3,2]-args.tip_shift \
  :dim[2]*1j])
end = time.time()
print("Reading tip positions in {} seconds for rank {}.".format(end-start, \
  rank))

# ------------------------------------------------------------------------------
# READ AND EVALUATE CHEN'S COEFFICIENTS
# ------------------------------------------------------------------------------

# Energy limits for tip
minETip = min(args.voltages)
maxETip = max(args.voltages)
chenSingles = []

start = time.time()
if rank == 0: # Functions called here only support single process!
  idx = 0 # Index of input argument
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
  chenCoeffs = ChenCoeffsPython(noOrbs=args.orbs_tip, singles=chenSingles, \
    eigs=eigsTip, rotate=False)
else:
  chenCoeffs = None
chenCoeffs = comm.bcast(chenCoeffs, root=0)
chenCoeffs.setGrids(tipPos)
end = time.time()
print("Reading tip coefficients in {} seconds for rank {}.".format(end-start, \
  rank))

# ------------------------------------------------------------------------------
# EVALUATE CP2K WAVE FUNCTION ON REGULAR GRID
# ------------------------------------------------------------------------------

# TODO determine extrapolation plane!

# TODO these need to be evaluated across process!
# z-interval + safety bounds
zmin = np.min(tipPos[-1][2])-args.dx
zmax = np.max(tipPos[1][2])+args.dx
evalRegion = np.array([[lVec[0,0],lVec[0,0]+lVec[1,0]], \
                       [lVec[0,1],lVec[0,1]+lVec[2,1]], \
                       [zmin,zmax]])
print("CP2K Evaluation Region:\n", evalRegion)

start = time.time()
# Set up CP2K wave function object
wfn = cgo.Cp2kGridOrbitals(rank, size, mpi_comm=comm)
wfn.read_cp2k_input(args.cp2k_input)
wfn.read_xyz(args.xyz)
wfn.center_atoms_to_cell()
wfn.read_basis_functions(args.basis_sets)
wfn.load_restart_wfn_file(args.coeffs,
  emin=args.emin-2.0*args.fwhm,
  emax=args.emax+2.0*args.fwhm)
wfn.calc_morbs_in_region(args.dx,
  x_eval_region=evalRegion[0]*ang2bohr,
  y_eval_region=evalRegion[1]*ang2bohr,
  z_eval_region=evalRegion[2]*ang2bohr,
  reserve_extrap = args.extrap_extent,
  eval_cutoff = args.rcut)
end = time.time()
print("Building CP2K wave function matrix in {} seconds for rank {}.".format( \
  end-start, rank))

# ------------------------------------------------------------------------------
# EXTRAPOLATE WAVE FUNCTION
# ------------------------------------------------------------------------------

start = time.time()
hartCube = Cube()
hartCube.read_cube_file(args.hartree_file)
extrapPlaneZ = evalRegion[2][1] - np.max(wfn.ase_atoms.positions[:,2])
hartPlane = hartCube.get_plane_above_topmost_atom(extrapPlaneZ) \
  - wfn.ref_energy*ev2hartree
wfn.extrapolate_morbs(hart_plane=hartPlane)
end = time.time()
print("Extrapolating CP2K wave function matrix in {} seconds for rank {}."\
  .format(end-start, rank))

# ------------------------------------------------------------------------------
# CREATE WAVE FUNCTION OBJECT (WRAPPER)
# ------------------------------------------------------------------------------

# TODO Parallelization! Use the tip position on each rank to determine the
#      necessary values from wave function matrix and then request it.
#      (Create a shared vector with information what each process needs, then
#       send the chunks according to this vector! (Note: Only along x-axis)
wfnSam = WavefunctionHelp(eigs=wfn.morb_energies, atoms=wfn.ase_atoms, \
  noOrbsTip=args.orbs_tip, wfnMatrix=wfn.morb_grids, evalRegion=evalRegion)
wfnSam.setGrids(tipPos[1:])

# ==============================================================================
# ------------------------------------------------------------------------------
#                             CREATE HR-STM OBJECT
# ------------------------------------------------------------------------------
# ==============================================================================

# TODO (use the python one, slow on this machine but I believe it's just as fast
#       on daint)

start = time.time()
hrstm = HRSTM(chenCoeffs, wfnSam, args.fwhm, \
  rank, size, comm)
del tipPos
hrstm.run(args.voltages)
end = time.time()
print("Evaluating HRSTM-run method in {} seconds.".format(end-start))

print(np.shape(hrstm.localCurrent))
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(hrstm.localCurrent[:,:,0,0])
plt.show()


endTotal = time.time()
print("Total time was {} seconds for rank {}.".format(endTotal-startTotal, \
  rank))
