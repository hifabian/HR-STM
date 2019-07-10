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
import time

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

from python.tunnelling.chen_coeffs_python import *

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
# ------------------------------------------------------------------------------

parser.add_argument('--output',
  type=str,
  metavar="OUTPUT",
  default="hrstm",
  required=False,
  help="Name for output file.")

# ------------------------------------------------------------------------------

parser.add_argument('--cp2k_input',
  metavar='FILE',
  required=True,
  help="CP2K input file used for the sample calculation.")

parser.add_argument('--basis_sets',
  metavar='FILE',
  required=True,
  help="File containing all basis sets used for sample.")

parser.add_argument('--xyz',
  metavar='FILE',
  required=True,
  help="File containing the atom positions for the sample.")

parser.add_argument('--coeffs',
  metavar='FILE',
  required=True,
  help="File containing the basis coefficients for the sample"
  + " (*.wfn or *.MOLog).")

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
  required=True,
  help="File paths to positions.")

parser.add_argument('--tip_shift',
  type=float,
  metavar='A',
  nargs="+",
  required=True,
  help="z-distance shift for the metal tip with respect to apex"
  + " in Angstrom.")

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
  required=True,
  help="Integer indicating which orbitals of the tip are used following the"
  + " angular momentum quantum number.")

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
  default=-2.0,
  help="Lower energy used for cutoff with respect to Fermi energy in eV"
  + " for sample.")

parser.add_argument('--emax',
  type=float,
  metavar='E',
  required=False,
  default=2.0,
  help="Upper energy used for cutoff with respect to Fermi energy in eV"
  + " for sample.")

parser.add_argument('--dx',
  type=float,
  default=0.1)

parser.add_argument('--extrap_extent',
  type=float,
  default=5.0)

parser.add_argument('--fwhm',
  type=float,
  default=0.05)

parser.add_argument('--hartree_file')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ==============================================================================
# ------------------------------------------------------------------------------
#                         READ FILES AND SET UP INPUT
# ------------------------------------------------------------------------------
# ==============================================================================

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

# TODO Currently only for 1 process, this should be distributed care has to be
#      taken since they may not align with distribution of wfn object!
#      No idea how to handle this as of yet...

tipPos = []
for filename in args.tip_pos:
  tmp, lVec = read_PPPos(filename)
  # Periodic boundaries along x- and y-direction
  tipPos.append(apply_bounds(tmp,lVec))
dim = np.shape(tipPos[0])[1:]
# Metal tip (needed for rotation, no tunnelling considered)
tipPos.insert(0, np.mgrid[ \
  lVec[0,0]:lVec[0,0]+lVec[1,0]:dim[0]*1j, \
  lVec[0,1]:lVec[0,1]+lVec[2,1]:dim[1]*1j, \
  lVec[0,2]-args.tip_shift[0]:lVec[0,2]+lVec[3,2]-args.tip_shift[0] \
  :dim[2]*1j])

# TODO Later, tipPos is hopefully already split of splitting here,
#      we don't split along z axis (K only does x I believe), so 
#      splitting before this should be fine
dx = lVec[1,0] / dim[0]
dy = lVec[2,1] / dim[1]
dz = lVec[3,2] / dim[2]
reg_step = np.min([dx,dy,dz])
# Last position correspond to closest tip atom
zmin = np.min(tipPos[-1][2])-dz
# Second position correspond to furthest relevant tip atom
zmax = np.max(tipPos[1][2])+dz
# NOTE Conversion to Bohr, done for other values at appropriate times,
#      but not for this. Thus, this is the only object in Bohr!
eval_reg = [[lVec[0,0]*ang2bohr,(lVec[0,0]+lVec[1,0])*ang2bohr],
            [lVec[0,1]*ang2bohr,(lVec[0,1]+lVec[2,1])*ang2bohr],
            [zmin*ang2bohr,zmax*ang2bohr]]
print(eval_reg)

# ------------------------------------------------------------------------------
# READ AND EVALUATE CHEN'S COEFFICIENTS
# ------------------------------------------------------------------------------

# Energy limits for tip
minETip = min(args.voltages)
maxETip = max(args.voltages)

chenSingles = []

start = time.time()
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

chenCoeffs = ChenCoeffsPython(noOrbs=args.orbs_tip, singles=chenSingles, \
  eigs=eigsTip, rotate=False)

# ------------------------------------------------------------------------------
# EVALUATE CP2K WAVE FUNCTION ON REGULAR GRID
# ------------------------------------------------------------------------------

wfn = cgo.Cp2kGridOrbitals(rank, size, mpi_comm=comm)
wfn.read_cp2k_input(args.cp2k_input)
wfn.read_xyz(args.xyz)
wfn.center_atoms_to_cell()
wfn.read_basis_functions(args.basis_sets)
wfn.load_restart_wfn_file(args.coeffs, 
  emin=args.emin-2.0*args.fwhm,
  emax=args.emax+2.0*args.fwhm)

print("R%d/%d: loaded wfn, %.2fs"%(rank, size, (time.time() - time0)))
sys.stdout.flush()
time1 = time.time()

wfn.calc_morbs_in_region(reg_step,
  x_eval_region=eval_reg[0],
  y_eval_region=eval_reg[1],
  z_eval_region=eval_reg[2],
  reserve_extrap = args.extrap_extent,
  eval_cutoff = args.rcut)

print("R%d/%d: evaluated wfn, %.2fs"%(rank, size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()

print(rank, np.shape(wfn.morb_grids))

# ------------------------------------------------------------------------------
# EXTRAPOLATE WAVE FUNCTION
# ------------------------------------------------------------------------------

hart_cube = Cube()
hart_cube.read_cube_file(args.hartree_file)
extrap_plane_z = eval_reg[2][1] / ang2bohr \
  - np.max(wfn.ase_atoms.positions[:, 2])
hart_plane = hart_cube.get_plane_above_topmost_atom(extrap_plane_z) \
  - wfn.ref_energy*ev2hartree
wfn.extrapolate_morbs(hart_plane=hart_plane)

print("R%d/%d: extrapolated wfn, %.2fs"%(rank, size, (time.time() - time1)))
sys.stdout.flush()
time1 = time.time()


# TODO interpolation should occur when evaluating tunnelling matrix
# Use wrapper class for this? maybe I can reuse/minimally adjust part of Master
# thesis code!
# ------------------------------------------------------------------------------
# INTERPOLATE WAVE FUNCTIONS
# ------------------------------------------------------------------------------

# TODO divide up wave function to processes before calling STM class (K handles
# it inside STM class... really have it outside?)


print(wfn.dv)
print(wfn.eval_cell_n)
for i in range(3):
  print((wfn.eval_cell_n[i]-1)*wfn.dv[i])
x = np.linspace(0,(wfn.eval_cell_n[0]-1)*wfn.dv[0],wfn.eval_cell_n[0])
y = np.linspace(0,(wfn.eval_cell_n[1]-1)*wfn.dv[1],wfn.eval_cell_n[0])
z = np.linspace(lVec[0,2],lVec[0,2]+lVec[3,2],wfn.eval_cell_n[0])

from scipy.interpolate import RegularGridInterpolator
linInp = RegularGridInterpolator((x,y,z),wfn.morb_grids[0][0], method="linear")
































