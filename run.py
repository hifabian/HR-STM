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

parser.add_argument('--rotate',
  action="store_true",
  default=False,
  required=False,
  help="If set, tip coefficients will be rotated.")

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

parser.add_argument('--extrap_dist',
  type=float,
  metavar='Ang',
  default=0.5,
  required=False,
  help="Starting distance from highest atom where extrapolation is used from.")

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
#      processes. Ideally this would be done while reading ;)


start = time.time()
if rank == 0:
  tipPosCmplt = []
  for filename in args.tip_pos:
    tmp, lVec = read_PPPos(filename)
    # Restrict tip positions to cell box and store it
#    tipPosCmplt.append(apply_bounds(tmp,lVec))
    tipPosCmplt.append(tmp)
  dim = np.shape(tipPosCmplt[0])[1:]
  # Metal tip (needed for rotation, no tunnelling considered)
  # NOTE Not needed after determining the rotational matrices!
  tipPosCmplt.insert(0, np.mgrid[ \
    lVec[0,0]:lVec[0,0]+lVec[1,0]:dim[0]*1j, \
    lVec[0,1]:lVec[0,1]+lVec[2,1]:dim[1]*1j, \
    lVec[0,2]-args.tip_shift:lVec[0,2]+lVec[3,2]-args.tip_shift \
    :dim[2]*1j])
  end = time.time()
  # TODO these need to be evaluated across process!
  # z-interval + safety bounds
  # TODO xmin, xmax, ymin, ymax should consider all relaxed grids...
  # I'm doing some re-computations with this evaluation region but it works
  # fine, makes the parallelisation simpler and even avoids some bugs!
  xmin = np.min(tipPosCmplt[-1][0])-args.dx
  xmax = np.max(tipPosCmplt[-1][0])+args.dx
  ymin = np.min(tipPosCmplt[-1][1])-args.dx
  ymax = np.max(tipPosCmplt[-1][1])+args.dx
  zmin = np.min(tipPosCmplt[-1][2])-args.dx
  zmax = np.max(tipPosCmplt[1][2])+args.dx
  evalRegion = np.array([[xmin,xmax], \
                         [ymin,ymax], \
                         [zmin,zmax]])
#  evalRegion = np.array([[lVec[0,0],lVec[0,0]+lVec[1,0]], \
#                         [lVec[0,1],lVec[0,1]+lVec[2,1]], \
#                         [zmin,zmax]])
else:
  dim = None
  lVec = None
  evalRegion = None
  tipPosCmplt = [[None]*3]*(len(args.tip_pos)+1)
# Broadcast total dimension and lVec
dim = comm.bcast(dim, root=0)
lVec = comm.bcast(lVec, root=0)
evalRegion = comm.bcast(evalRegion, root=0)
# x-indices on each rank
allXIds = np.array_split(np.arange(dim[0]), size)
# Information on split sizes
lengths = [len(allXIds[rank])*dim[1]*dim[2] for rank in range(size)]
offsets = [allXIds[rank][0]*dim[1]*dim[2] for rank in range(size)]
# Storage
dimLocal = (len(allXIds[rank]),)+dim[1:]
tipPos = [np.empty((3,)+dimLocal) for idx in range(len(args.tip_pos)+1)]
for posIdx in range(len(args.tip_pos)+1):
  # Split tip positions for (x,y,z) separately
  for axis in range(3):
    comm.Scatterv([tipPosCmplt[posIdx][axis], lengths, offsets, MPI.DOUBLE], \
      tipPos[posIdx][axis], root=0)

end = time.time()
print("Reading tip positions in {} seconds for rank {}.".format(end-start, \
  rank))

print(rank, np.min(tipPos[0][0]), np.max(tipPos[0][0]), np.mean(tipPos[0][0]), np.shape(tipPos[0][0]))
print(rank, np.min(tipPos[2][0]), np.max(tipPos[1][0]), np.mean(tipPos[1][0]), np.shape(tipPos[1][0]))
print(rank, np.min(tipPos[1][0]), np.max(tipPos[2][0]), np.mean(tipPos[2][0]), np.shape(tipPos[2][0]))

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
      chenSingle, eigsTip = const_coeffs(minETip, maxETip,
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
    eigs=eigsTip, rotate=args.rotate)
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
#args.extrap_dist
args.extrap_extent = 0
  

print("CP2K Evaluation Region:\n", evalRegion)


start = time.time()
# Set up CP2K wave function object
wfn = cgo.Cp2kGridOrbitals(rank, size, mpi_comm=comm)
wfn.read_cp2k_input(args.cp2k_input)
wfn.read_xyz(args.xyz)
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
dv = wfn.dv / ang2bohr
print(wfn.dv, dv)
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

# TODO Just do it here for now, then pack it into a function.
# TODO I need to, determine which rank needs which region and from whom it gets
#      which orbitals.
#      For the dimension, the easiest thing to do is look at the wfnMatrix size,
#      while the grid is defined by evalRegion. What is needed by which rank is
#      given through xmin, xmax evaluated locally (z and y are complete!).
#      Restrict xmin, xmax to region should give the index (floor, ceil).
#      For the energies, first assemble all energies bla bla (do what K does)
eigsSam = []
wfnMatrix = []
ase_atoms = wfn.ase_atoms
for spinIdx in range(wfn.nspin):
  # Gather eigen energies of sample
  tmp = comm.allgather(wfn.morb_energies[spinIdx])
  noEigsByRank = np.array([len(val) for val in tmp])
  eigsSam.append(np.hstack(tmp))
  del tmp

  # Indices needed for the tip position on this process
  xIds = np.array([ \
    np.floor((np.min(tipPos[-1][0])-evalRegion[0][0]) / dv[0]), \
    np.ceil((np.max(tipPos[-1][0])-evalRegion[0][0]) / dv[0])], \
    dtype=int)
  xIdsAll = comm.allgather(xIds)
  # Dimension of local grid for wave function matrix
  wfnDimLocal = (xIds[1]-xIds[0]+1,)+np.shape(wfn.morb_grids[spinIdx])[2:]
  noPoints = np.product(wfnDimLocal)
  print(rank, noPoints, wfnDimLocal)
  # Gather the necessary stuff
  for r in range(size):
    if rank == r:
      recvbuf = np.empty(len(eigsSam[spinIdx])*noPoints, dtype=wfn.morb_grids[spinIdx][spinIdx].dtype)
    else:
      recvbuf = None
    sendbuf = wfn.morb_grids[spinIdx][:,xIdsAll[r][0]:xIdsAll[r][1]+1].ravel()
    comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf,noEigsByRank*noPoints], root=r)
    if rank == r:
      wfnMatrix.append(recvbuf.reshape((len(eigsSam[spinIdx]),)+wfnDimLocal))
# Free memory by deleting large unnecessary things
del wfn
# Divide grids by space rather than by eigen energies, then free some memory
#del wfn
# TODO Parallelization! Use the tip position on each rank to determine the
#      necessary values from wave function matrix and then request it.
#      (Create a shared vector with information what each process needs, then
#       send the chunks according to this vector! (Note: Only along x-axis)
# TODO let's move this into HRSTM eventually!
# TODO evalRegion is incorrect with the parallelization
evalRegionLocal = evalRegion
evalRegionLocal[0] = evalRegion[0][0]+xIds*dv[0]
wfnSam = WavefunctionHelp(eigs=eigsSam, atoms=ase_atoms, \
  noOrbsTip=args.orbs_tip, wfnMatrix=wfnMatrix, evalRegion=evalRegionLocal)
wfnSam.setGrids(tipPos[1:])

# ==============================================================================
# ------------------------------------------------------------------------------
#                             CREATE HR-STM OBJECT
# ------------------------------------------------------------------------------
# ==============================================================================

# TODO (use the python one, slow on this machine but I believe it's just as fast
#       on daint)
if rank == 0:
  # Meta information
  meta = {'dimGrid' : np.shape(tipPos[0])[:-1], \
          'lVec' : lVec, \
          'voltages' : args.voltages, \
  }
  np.save(args.output+"_meta.npy", meta)

start = time.time()
hrstm = HRSTM(chenCoeffs, wfnSam, args.fwhm, \
  rank, size, comm, dim)
del tipPos
hrstm.run(args.voltages)
current = hrstm.gather()
if rank == 0:
  np.savez_compressed(args.output, current.ravel())

end = time.time()
print("Evaluating HRSTM-run method in {} seconds.".format(end-start))

print(np.shape(hrstm.localCurrent))
if rank == 0:
  import matplotlib.pyplot as plt
  plt.figure()
  plt.imshow(current[:,:,3,0], cmap="gist_gray")
  plt.show()
