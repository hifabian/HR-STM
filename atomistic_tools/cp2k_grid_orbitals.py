"""
Tools to put CP2K orbitals on a real space grid
""" 

import os
import numpy as np
import scipy
import scipy.io
import time
import copy
import sys

import re
import io
import ase
import ase.io

from .cube import Cube
from .cp2k_wfn_file import Cp2kWfnFile

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

class Cp2kGridOrbitals:
    """
    Class to load and put CP2K orbitals on a discrete real-space grid.
    The orbitals will be equally divided between the mpi processes.
    """
    
    def __init__(self, mpi_rank=0, mpi_size=1, mpi_comm=None, single_precision=True):

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        
        if single_precision:
            self.dtype = np.float32
        else:
            self.dtype = np.float64

        # geometry
        self.cell = None # Bohr radii / [au]
        self.ase_atoms = None 

        # Basis set
        self.elem_basis_name = None # basis set names we are using for elements
        self.basis_sets = None

        # The global energy limits when loading the orbitals
        self.emin = None
        self.emax = None

        # Object to deal with loading molecular orbitals from .wfn file
        self.cwf = Cp2kWfnFile(self.mpi_rank, self.mpi_size, self.mpi_comm)

        # Set by cwf:
        self.morb_composition = None
        self.morb_energies = None
        self.i_homo_loc = None
        self.nspin = None
        self.ref_energy = None
        self.global_morb_energies = None

        # Orbitals on discrete grid
        self.morb_grids = None
        self.dv = None # [dx, dy, dz] in [au]
        self.origin = None
        self.eval_cell = None
        self.eval_cell_n = None

        self.last_calc_iz = None # last directly calculated z plane (others extrapolated)
        


    ### -----------------------------------------
    ### General cp2k routines
    ### -----------------------------------------

    def read_cp2k_input(self, cp2k_input_file):
        """
        Reads from the cp2k input file:
        * Basis set names for all elements
        * Cell size
        """
        self.elem_basis_name = {}
        self.cell = np.zeros(3)
        with open(cp2k_input_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                parts = lines[i].split()
                if len(parts) == 0:
                    continue
                # Have we found the basis set info?
                if parts[0] == "&KIND":
                    elem = parts[1]
                    for j in range(10):
                        parts = lines[i+j].split()
                        if parts[0] == "BASIS_SET":
                            basis = parts[1]
                            self.elem_basis_name[elem] = basis
                            break
                # Have we found the CELL info?
                if parts[0] == "ABC":   
                    if parts[1] == "[angstrom]":
                        self.cell[0] = float(parts[2])
                        self.cell[1] = float(parts[3])
                        self.cell[2] = float(parts[4])
                    else:
                        self.cell[0] = float(parts[1])
                        self.cell[1] = float(parts[2])
                        self.cell[2] = float(parts[3])

                if parts[0] == "A" or parts[0] == "B" or parts[0] == "C":
                    prim_vec = np.array([float(x) for x in parts[1:]])
                    if np.sum(prim_vec > 0.0) > 1:
                        raise ValueError("Cell is not rectangular")
                    ind = np.argmax(prim_vec > 0.0)
                    self.cell[ind] = prim_vec[ind]

        self.cell *= ang_2_bohr

        if any(self.cell < 1e-3):
            raise ValueError("Cell " + str(self.cell) + " is invalid")

        if self.ase_atoms is not None:
            self.ase_atoms.cell = self.cell / ang_2_bohr

    def read_xyz(self, file_xyz):
        """ Read atomic positions from .xyz file (in Bohr radiuses) """
        with open(file_xyz) as f:
            fxyz_contents = f.read()
        # Replace custom elements (e.g. for spin-pol calcs)
        fxyz_contents = re.sub("([a-zA-Z]+)[0-9]+", r"\1", fxyz_contents)
        self.ase_atoms = ase.io.read(io.StringIO(fxyz_contents), format="xyz")

        if self.cell is not None:
            self.ase_atoms.cell = self.cell / ang_2_bohr

    def center_atoms_to_cell(self):
        self.ase_atoms.center()
            
    ### -----------------------------------------
    ### Basis set routines
    ### -----------------------------------------

    def _magic_basis_normalization(self, basis_sets_):
        """ Normalizes basis sets to be compatible with cp2k """
        basis_sets = copy.deepcopy(basis_sets_)
        for elem, bsets in basis_sets.items():
            for bset in bsets:
                for shell in bset:
                    l = shell[0]
                    exps = shell[1]
                    coefs = shell[2]
                    nexps = len(exps)

                    norm_factor = 0
                    for i in range(nexps-1):
                        for j in range(i+1, nexps):
                            norm_factor += 2*coefs[i]*coefs[j]*(2*np.sqrt(exps[i]*exps[j])/(exps[i]+exps[j]))**((2*l+3)/2)

                    for i in range(nexps):
                        norm_factor += coefs[i]**2

                    for i in range(nexps):
                        coefs[i] = coefs[i]*exps[i]**((2*l+3)/4)/np.sqrt(norm_factor)

        return basis_sets

    def read_basis_functions(self, basis_set_file):
        """ Reads the basis sets from basis_set_file specified in elem_basis_name

        returns:
        basis_sets["Element"] = 
        """
        self.basis_sets = {}
        with open(basis_set_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                parts = lines[i].split()
                if len(parts) == 0:
                    continue
                if parts[0] in self.elem_basis_name:
                    elem = parts[0]
                    if parts[1] == self.elem_basis_name[elem] or (len(parts) > 2 and parts[2] == self.elem_basis_name[elem]):
                        # We have found the correct basis set
                        basis_functions = []
                        nsets = int(lines[i+1])
                        cursor = 2
                        for j in range(nsets):
                            
                            basis_functions.append([])

                            comp = [int(x) for x in lines[i+cursor].split()]
                            n_princ, l_min, l_max, n_exp = comp[:4]
                            l_arr = np.arange(l_min, l_max+1, 1)
                            n_basisf_for_l = comp[4:]
                            assert len(l_arr) == len(n_basisf_for_l)

                            exps = []
                            coeffs = []

                            for k in range(n_exp):
                                exp_c = [float(x) for x in lines[i+cursor+k+1].split()]
                                exps.append(exp_c[0])
                                coeffs.append(exp_c[1:])

                            exps = np.array(exps)
                            coeffs = np.array(coeffs)

                            indx = 0
                            for l, nl in zip(l_arr, n_basisf_for_l):
                                for il in range(nl):
                                    basis_functions[-1].append([l, exps, coeffs[:, indx]])
                                    indx += 1
                            cursor += n_exp + 1

                        self.basis_sets[elem] = basis_functions

        self.basis_sets = self._magic_basis_normalization(self.basis_sets)


    ### -----------------------------------------
    ### WFN file routines
    ### -----------------------------------------

    def load_restart_wfn_file(self, restart_file, emin=None, emax=None, n_homo=None, n_lumo=None):
        """
        Reads the specified molecular orbitals from cp2k restart wavefunction file
        If both, energy limits and counts are given, then the extreme is used
        Note that the energy range is in eV and with respect to HOMO energy.
        """

        self.cwf.load_restart_wfn_file(restart_file, emin=emin, emax=emax, n_homo=n_homo, n_lumo=n_lumo)
        self.cwf.convert_readable()

        self.morb_composition = self.cwf.morb_composition
        self.morb_energies = self.cwf.morb_energies
        self.i_homo_loc = self.cwf.i_homo_loc
        self.nspin = self.cwf.nspin
        self.ref_energy = self.cwf.ref_energy
        self.global_morb_energies = self.cwf.glob_morb_energies


    ### ---------------------------------------------------------------------------
    ### Methods directly related to putting stuff on grids
    ### ---------------------------------------------------------------------------

    def _spherical_harmonic_grid(self, l, m, x_grid, y_grid, z_grid):
        """
        Evaluates the spherical harmonics (times r^l) with some unknown normalization
        (source: Carlo's Fortran code)
        """
        c = (2.0/np.pi)**(3.0/4.0)

        # s orbitals
        if (l, m) == (0, 0):
            return c

        # p orbitals
        elif (l, m) == (1, -1):
            return c*2.0*y_grid
        elif (l, m) == (1, 0):
            return c*2.0*z_grid
        elif (l, m) == (1, 1):
            return c*2.0*x_grid

        # d orbitals
        elif (l, m) == (2, -2):
            return c*4.0*x_grid*y_grid
        elif (l, m) == (2, -1):
            return c*4.0*y_grid*z_grid
        elif (l, m) == (2, 0):
            return c*2.0/np.sqrt(3)*(2*z_grid**2-x_grid**2-y_grid**2)
        elif (l, m) == (2, 1):
            return c*4.0*z_grid*x_grid
        elif (l, m) == (2, 2):
            return c*2.0*(x_grid**2-y_grid**2)

        # f orbitals
        elif (l, m) == (3, -3):
            return c*np.sqrt(8/3)*y_grid*(3*x_grid**2-y_grid**2)
        elif (l, m) == (3, -2):
            return c*8.0*x_grid*y_grid*z_grid
        elif (l, m) == (3, -1):
            return c*np.sqrt(8/5)*y_grid*(4*z_grid**2-x_grid**2-y_grid**2)
        elif (l, m) == (3, 0):
            return c*4.0/np.sqrt(15.0)*z_grid*(2.0*z_grid**2-3.0*x_grid**2-3.0*y_grid**2)
        elif (l, m) == (3, 1):
            return c*np.sqrt(8/5)*x_grid*(4*z_grid**2-x_grid**2-y_grid**2)
        elif (l, m) == (3, 2):
            return c*4.0*z_grid*(x_grid**2-y_grid**2)
        elif (l, m) == (3, 3):
            return c*np.sqrt(8/3)*x_grid*(x_grid**2-3.0*y_grid**2)

        print("No spherical harmonic found for l=%d, m=%d" % (l, m))
        return 0


    def _add_local_to_global_grid(self, loc_grid, glob_grid, origin_diff, wrap=(True, True, True)):
        """
        Method to add a grid to another one
        Arguments:
        loc_grid -- grid that will be added to the glob_grid
        glob_grid -- defines "wrapping" boundaries
        origin_diff -- difference of origins between the grids; ignored for directions without wrapping
        wrap -- specifies in which directions to wrap and take PBC into account
        """
        loc_n = np.shape(loc_grid)
        glob_n = np.shape(glob_grid)
        od = origin_diff

        inds = []
        l_inds = []

        for i in range(len(glob_n)):
            
            if wrap[i]:
                # Move the origin_diff vector to the main global cell if wrapping is enabled
                od[i] = od[i] % glob_n[i]

                ixs = [[od[i], od[i] + loc_n[i]]]
                l_ixs = [0]
                while ixs[-1][1] > glob_n[i]:
                    overshoot = ixs[-1][1]-glob_n[i]
                    ixs[-1][1] = glob_n[i]
                    l_ixs.append(l_ixs[-1]+glob_n[i]-ixs[-1][0])
                    ixs.append([0, overshoot])
                l_ixs.append(loc_n[i])

                inds.append(ixs)
                l_inds.append(l_ixs)
            else:
                inds.append([-1])
                l_inds.append([-1])

        l_ixs = l_inds[0]
        l_iys = l_inds[1]
        l_izs = l_inds[2]
        for i, ix in enumerate(inds[0]):
            for j, iy in enumerate(inds[1]):
                for k, iz in enumerate(inds[2]):
                    if wrap[0]:
                        i_gl_x = slice(ix[0], ix[1])
                        i_lc_x = slice(l_ixs[i], l_ixs[i+1])
                    else:
                        i_gl_x = slice(None)
                        i_lc_x = slice(None)
                    if wrap[1]:
                        i_gl_y = slice(iy[0], iy[1])
                        i_lc_y = slice(l_iys[j], l_iys[j+1])
                    else:
                        i_gl_y = slice(None)
                        i_lc_y = slice(None)
                    if wrap[2]:
                        i_gl_z = slice(iz[0], iz[1])
                        i_lc_z = slice(l_izs[k], l_izs[k+1])
                    else:
                        i_gl_z = slice(None)
                        i_lc_z = slice(None)
                    
                    glob_grid[i_gl_x, i_gl_y, i_gl_z] += loc_grid[i_lc_x, i_lc_y, i_lc_z]


    def calc_morbs_in_region(self, dr_guess,
                            x_eval_region = None,
                            y_eval_region = None,
                            z_eval_region = None,
                            eval_cutoff = 14.0,
                            reserve_extrap = 0.0,
                            print_info = True):
        """ 
        Puts the molecular orbitals onto a specified grid
        Arguments:
        dr_guess -- spatial discretization step [ang], real value will change for every axis due to rounding  
        x_eval_region -- x evaluation (min, max) in [au]. If min == max, then evaluation only works on a plane.
                        If set, no PBC applied in direction and also no eval_cutoff.
        eval_cutoff -- cutoff in [ang] for orbital evaluation if eval_region is None
        """

        time1 = time.time()

        dr_guess *= ang_2_bohr
        eval_cutoff *= ang_2_bohr
        reserve_extrap *= ang_2_bohr

        global_cell_n = (np.round(self.cell/dr_guess)).astype(int)
        self.dv = self.cell / global_cell_n

        # Define local grid for orbital evaluation
        # and convenient PBC implementation
        eval_regions = [x_eval_region, y_eval_region, z_eval_region]
        loc_cell_arrays = []
        mid_ixs = np.zeros(3, dtype=int)
        loc_cell_n = np.zeros(3, dtype=int)
        eval_cell_n = np.zeros(3, dtype=int)
        self.origin = np.zeros(3)
        for i in range(3):
            if eval_regions[i] is None:
                # Define range in i direction with 0.0 at index mid_ixs[i]
                loc_arr = np.arange(0, eval_cutoff, self.dv[i])
                mid_ixs[i] = int(len(loc_arr)/2)
                loc_arr -= loc_arr[mid_ixs[i]]
                loc_cell_arrays.append(loc_arr)
                eval_cell_n[i] = global_cell_n[i]
                self.origin[i] = 0.0
            else:
                # Define the specified range in direction i
                v_min, v_max = eval_regions[i]
                ### TODO: Probably should use np.arange to have exactly matching dv in the local grid... ###
                loc_cell_arrays.append(np.linspace(v_min, v_max, int(np.round((v_max-v_min)/self.dv[i]))+1))
                mid_ixs[i] = -1
                eval_cell_n[i] = len(loc_cell_arrays[i])
                self.origin[i] = v_min
                
            loc_cell_n[i] = len(loc_cell_arrays[i])

        loc_cell_grids = np.meshgrid(loc_cell_arrays[0], loc_cell_arrays[1], loc_cell_arrays[2], indexing='ij')

        # Some info
        if print_info:
            print("Global cell: ", global_cell_n)
            print("Eval cell: ", eval_cell_n)
            print("local cell: ", loc_cell_n)
            print("---- Setup: %.4f" % (time.time() - time1))

        time_radial_calc = 0.0
        time_spherical = 0.0
        time_loc_glob_add = 0.0
        time_loc_lmorb_add = 0.0

        nspin = len(self.morb_composition)

        num_morbs = []
        morb_grids_local = []
        self.morb_grids = []

        ext_z_n = int(np.round(reserve_extrap/self.dv[2]))

        for ispin in range(nspin):
            num_morbs.append(len(self.morb_composition[ispin][0][0][0][0]))
            self.morb_grids.append(np.zeros((num_morbs[ispin], eval_cell_n[0], eval_cell_n[1], eval_cell_n[2] + ext_z_n), dtype=self.dtype))
            morb_grids_local.append(np.zeros((num_morbs[ispin], loc_cell_n[0], loc_cell_n[1], loc_cell_n[2]), dtype=self.dtype))

        self.eval_cell_n = np.array([eval_cell_n[0], eval_cell_n[1], eval_cell_n[2] + ext_z_n])
        self.eval_cell = self.eval_cell_n * self.dv
        self.last_calc_iz = eval_cell_n[2] - 1

        for i_at in range(len(self.ase_atoms)):
            elem = self.ase_atoms[i_at].symbol
            pos = self.ase_atoms[i_at].position * ang_2_bohr

            # how does the position match with the grid?
            int_shift = (pos/self.dv).astype(int)
            frac_shift = pos/self.dv - int_shift
            origin_diff = int_shift - mid_ixs

            # Shift the local grid such that origin is on the atom
            rel_loc_cell_grids = []
            for i, loc_grid in enumerate(loc_cell_grids):
                if eval_regions[i] is None:
                    rel_loc_cell_grids.append(loc_grid - frac_shift[i]*self.dv[i])
                else:
                    rel_loc_cell_grids.append(loc_grid - pos[i])

            r_vec_2 = rel_loc_cell_grids[0]**2 + \
                    rel_loc_cell_grids[1]**2 + \
                    rel_loc_cell_grids[2]**2

            for ispin in range(nspin):
                morb_grids_local[ispin].fill(0.0)

            for i_set, bset in enumerate(self.basis_sets[elem]):
                for i_shell, shell in enumerate(bset):
                    l = shell[0]
                    es = shell[1]
                    cs = shell[2]

                    # Calculate the radial part of the atomic orbital
                    time2 = time.time()
                    radial_part = np.zeros(loc_cell_n)
                    for e, c in zip(es, cs):
                        radial_part += c*np.exp(-1.0*e*r_vec_2)
                    time_radial_calc += time.time() - time2

                    for i_orb, m in enumerate(range(-l, l+1, 1)):
                        time2 = time.time()
                        atomic_orb = radial_part*self._spherical_harmonic_grid(l, m,
                                                                        rel_loc_cell_grids[0],
                                                                        rel_loc_cell_grids[1],
                                                                        rel_loc_cell_grids[2])
                        time_spherical += time.time() - time2
                        time2 = time.time()

                        for i_spin in range(nspin):
                            #print("---------------")
                            #print(i_spin, len(self.morb_composition))
                            #print(i_at, len(self.morb_composition[i_spin]))
                            #print(i_set, len(self.morb_composition[i_spin][i_at]))
                            #print(i_shell, len(self.morb_composition[i_spin][i_at][i_set]))
                            #print(i_orb, len(self.morb_composition[i_spin][i_at][i_set][i_shell]))
                            #print("---------------")

                            coef_arr = self.morb_composition[i_spin][i_at][i_set][i_shell][i_orb]

                            for i_mo in range(num_morbs[i_spin]):
                                morb_grids_local[i_spin][i_mo] += coef_arr[i_mo]*atomic_orb

                            # slow:
                            #morb_grids_local += np.outer(coef_arr, atomic_orb).reshape(
                            #                 num_morbs, loc_cell_n[0], loc_cell_n[1], loc_cell_n[2])
                        time_loc_lmorb_add += time.time() - time2

            time2 = time.time()
            for i_spin in range(nspin):
                for i_mo in range(num_morbs[i_spin]):
                    if ext_z_n == 0:
                        self._add_local_to_global_grid(
                            morb_grids_local[i_spin][i_mo],
                            self.morb_grids[i_spin][i_mo],
                            origin_diff,
                            wrap=(mid_ixs != -1))
                    else:
                        self._add_local_to_global_grid(
                            morb_grids_local[i_spin][i_mo],
                            self.morb_grids[i_spin][i_mo][:, :, :-ext_z_n],
                            origin_diff,
                            wrap=(mid_ixs != -1))
            time_loc_glob_add += time.time() - time2

        if print_info:
            print("---- Radial calc time : %4f" % time_radial_calc)
            print("---- Spherical calc time : %4f" % time_spherical)
            print("---- Loc -> loc_morb time : %4f" % time_loc_lmorb_add)
            print("---- loc_morb -> glob time : %4f" % time_loc_glob_add)
            print("---- Total time: %.4f"%(time.time() - time1))


    ### -----------------------------------------
    ### Extrapolate wavefunctions
    ### -----------------------------------------

    def _resize_2d_arr_with_interpolation(self, array, new_shape):
        x_arr = np.linspace(0, 1, array.shape[0])
        y_arr = np.linspace(0, 1, array.shape[1])
        rgi = scipy.interpolate.RegularGridInterpolator(points=[x_arr, y_arr], values=array)

        x_arr_new = np.linspace(0, 1, new_shape[0])
        y_arr_new = np.linspace(0, 1, new_shape[1])
        x_coords = np.repeat(x_arr_new, len(y_arr_new))
        y_coords = np.tile(y_arr_new, len(x_arr_new))

        return rgi(np.array([x_coords, y_coords]).T).reshape(new_shape)

    def extrapolate_morbs(self, vacuum_pot=None, hart_plane=None, use_weighted_avg=True):
        for ispin in range(self.nspin):
            self.extrapolate_morbs_spin(ispin, vacuum_pot=vacuum_pot, hart_plane=hart_plane, use_weighted_avg=use_weighted_avg)

    def extrapolate_morbs_spin(self, ispin, vacuum_pot=None, hart_plane=None, use_weighted_avg=True):
        """
        Extrapolate molecular orbitals from a specified plane to a box or another plane
        in case of "single_plane = True", the orbitals will be only extrapolated on
        a plane "extent" distance away
        Extent in bohr !!!

        Either the vacuum potential or the hartree plane is needed!
        Both are assumed to be in hartree units wrt to Fermi/Homo.

        NB: everything in hartree units!
        """
        time1 = time.time()

        if vacuum_pot is None and hart_plane is None:
            print("You must specify either the vac pot or the hartree plane.")
            return None

        morb_planes = self.morb_grids[ispin][:, :, :, self.last_calc_iz]
        morb_energies = self.morb_energies[ispin]

        num_morbs = np.shape(morb_planes)[0]

        for morb_index in range(num_morbs):

            morb_plane = morb_planes[morb_index]

            if vacuum_pot != None:
                hartree_avg = vacuum_pot
            else:
                if use_weighted_avg:
                    # weigh the hartree potential by the molecular orbital
                    density_plane = morb_plane**2
                    density_plane /= np.sum(density_plane)
                    weighted_hartree = density_plane * self._resize_2d_arr_with_interpolation(hart_plane, density_plane.shape)
                    hartree_avg = np.sum(weighted_hartree)
                else:
                    hartree_avg = np.mean(hart_plane)

            energy = morb_energies[morb_index]/hart_2_ev
            if energy > hartree_avg:
                print("Warning: unbound state, can't extrapolate! index: %d. Exiting." % morb_index)
                break

            fourier = np.fft.rfft2(morb_plane)
            # NB: rfft2 takes REAL fourier transform over last (y) axis and COMPLEX over other (x) axes
            # dv in BOHR, so k is in 1/bohr
            kx_arr = 2*np.pi*np.fft.fftfreq(morb_plane.shape[0], self.dv[0])
            ky_arr = 2*np.pi*np.fft.rfftfreq(morb_plane.shape[1], self.dv[1])

            kx_grid, ky_grid = np.meshgrid(kx_arr, ky_arr,  indexing='ij')

            prefactors = np.exp(-np.sqrt(kx_grid**2 + ky_grid**2 - 2*(energy - hartree_avg))*self.dv[2])
            for iz in range(self.last_calc_iz + 1, self.eval_cell_n[2]):
                fourier *= prefactors
                self.morb_grids[ispin][morb_index, :, :, iz] = np.fft.irfft2(fourier, morb_plane.shape)

        print("Extrapolation time: %.3f s"%(time.time()-time1))

    ### -----------------------------------------
    ### Export data
    ### -----------------------------------------

    def write_cube(self, filename, orbital_nr, spin=0, square=False):
        local_ind = self.i_homo_loc[spin] - orbital_nr
        if local_ind >= 0 and local_ind < self.morb_grids[spin].shape[0]:
            print("R%d/%d is writing HOMO%+d cube" %(self.mpi_rank, self.mpi_size, orbital_nr))
            if not square:
                c = Cube(title="HOMO%+d"%orbital_nr, comment="cube", ase_atoms=self.ase_atoms,
                    origin=self.origin, cell=self.eval_cell*np.eye(3), data=self.morb_grids[spin][local_ind])
            else:
                c = Cube(title="HOMO%+d square"%orbital_nr, comment="cube", ase_atoms=self.ase_atoms,
                    origin=self.origin, cell=self.eval_cell*np.eye(3), data=self.morb_grids[spin][local_ind]**2)
            c.write_cube_file(filename)
    
    def _orb_plane_above_atoms(self, grid, height):
        """
        Returns the 2d plane above topmost atom in z direction
        height in [angstrom]
        """
        topmost_atom_z = np.max(self.ase_atoms.positions[:, 2]) # Angstrom
        plane_z = (height + topmost_atom_z) * ang_2_bohr
        plane_z_wrt_orig = plane_z - self.origin[2]

        plane_index = int(np.round(plane_z_wrt_orig/self.eval_cell[2]*self.eval_cell_n[2]))
        return grid[:, :, plane_index]

    def collect_and_save_ch_orbitals(self, orbital_list, height_list, path = "./orb.npz"):
        """
        Save constant-height planes of selected orbitals at selected heights
        orbital list wrt to HOMO
        """
        slice_list = []

        for i_spin in range(self.nspin):
            slice_list.append([])
            for h in height_list:
                slice_list[i_spin].append([])

            for i_mo in range(len(self.morb_energies[i_spin])):
                i_mo_wrt_homo = i_mo - self.i_homo_loc[i_spin]

                if i_mo_wrt_homo in orbital_list:
                    for i_h, h in enumerate(height_list):
                        orb_plane = self._orb_plane_above_atoms(self.morb_grids[i_spin][i_mo], h)
                        slice_list[i_spin][i_h].append(orb_plane)
         
        # indexes of the slice_list: [i_spin], [i_h], [i_mo], [nx x ny]
        # gather to rank_0
        final_list = []

        for i_spin in range(self.nspin):
            final_list.append([]) # i_spin
            for i_h in range(len(height_list)):
                plane_gather = self.mpi_comm.gather(slice_list[i_spin][i_h], root = 0)

                if self.mpi_rank == 0:
                    # flatten the list of lists to numpy
                    flat_array = np.array([item for sublist in plane_gather for item in sublist])
                    final_list[i_spin].append(flat_array)        

        # select energy ranges
        #self.gather_global_energies()

        if self.mpi_rank == 0:
            # energy array
            # for rank 0, the homo index is given by loc_homo_ind
            save_energy_arr = []
            for i_spin in range(self.nspin):
                global_orb_list = [ind + self.i_homo_loc[i_spin] for ind in orbital_list]
                save_energy_arr.append(self.global_morb_energies[i_spin][global_orb_list])
            
            # turn the spin and height dimensions to numpy as well
            final_numpy = np.array([np.array(list_h) for list_h in final_list])
            save_data = {}
            save_data['orbitals'] = final_numpy
            save_data['heights'] = np.array(height_list)
            save_data['orb_list'] = np.array(orbital_list)
            save_data['x_arr'] = np.arange(0.0, self.eval_cell_n[0]*self.dv[0] + self.dv[0]/2, self.dv[0]) + self.origin[0]
            save_data['y_arr'] = np.arange(0.0, self.eval_cell_n[1]*self.dv[1] + self.dv[1]/2, self.dv[1]) + self.origin[1]
            save_data['energies'] = np.array(save_energy_arr)
            np.savez_compressed(path, **save_data)


    ### -----------------------------------------
    ### mpi communication
    ### -----------------------------------------

    #def gather_global_energies(self):
    #    self.global_morb_energies = []
    #    for ispin in range(self.nspin):
    #        morb_en_gather = self.mpi_comm.allgather(self.morb_energies[ispin])
    #        self.global_morb_energies.append(np.hstack(morb_en_gather))
