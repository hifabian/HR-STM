# @author Hillebrand, Fabian
# @date   2019

import numpy as np
import scipy as sp
import scipy.io

import time
import copy
import sys

import re
import io
import ase
import ase.io

from .cp2k_wfn_file import Cp2kWfnFile

ang_2_bohr = 1.0/0.52917721067
hart_2_ev = 27.21138602

class PPSTMGridOrbitals:
    """
    Class to load CP2K coefficients and put PPSTM orbitals on a discrete
    real-space grid.
    The orbitals are divided by energy equally among the MPI processes.
    """
    
    def __init__(self, mpi_rank=0, mpi_size=1, mpi_comm=None, single_precision=True):
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm
        if single_precision:
            self.dtype = np.float32
        else:
            self.dtype = np.float64
        # Geometry
        self.cell = None
        self.ase_atoms = None
        # CP2K Basis
        self.elem_basis_name = None
        self.basis_sets = None
        # Energy limits
        self.emin = None
        self.emax = None
        # CP2K wfn coeffficients
        self.cwf = Cp2kWfnFile(self.mpi_rank, self.mpi_size, self.mpi_comm)
        # CP2K wfn values
        self.morb_composition = None
        self.morb_energies = None
        self.i_homo_loc = None
        self.nspin = None
        self.ref_energy = None
        self.global_morb_energies = None
        # Orbitals and discrete grid
        self.morb_grids = None
        self.dv = None
        self.eval_cell = None
        self.eval_cell_n = None

    ### ------------------------------------------------------------------------
    ### General CP2K routines
    ### ------------------------------------------------------------------------

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

    ### -----------------------------------------
    ### Basis set routines
    ### -----------------------------------------

    def _magic_basis_normalization(self, basis_sets_):
        """ Normalizes basis sets to be compatible with cp2k """
        # TODO
        return basis_sets_

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


    ### ------------------------------------------------------------------------
    ### WFN file routines
    ### ------------------------------------------------------------------------

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


    ### ------------------------------------------------------------------------
    ### Methods directly related to putting stuff on grids
    ### ------------------------------------------------------------------------

    def _spherical_harmonic_grid(self, l, m, x_grid, y_grid, z_grid, r):
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
            return c*2.0*y_grid / r
        elif (l, m) == (1, 0):
            return c*2.0*z_grid / r 
        elif (l, m) == (1, 1):
            return c*2.0*x_grid / r

        # d orbitals
        elif (l, m) == (2, -2):
            return c*4.0*x_grid*y_grid / (r**2)
        elif (l, m) == (2, -1):
            return c*4.0*y_grid*z_grid / (r**2)
        elif (l, m) == (2, 0):
            return c*2.0/np.sqrt(3)*(2*z_grid**2-x_grid**2-y_grid**2) / (r**2)
        elif (l, m) == (2, 1):
            return c*4.0*z_grid*x_grid / (r**2)
        elif (l, m) == (2, 2):
            return c*2.0*(x_grid**2-y_grid**2) / (r**2)

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
        Puts the molecular orbitals onto a specified grid.

        @param dr_guess       Spatial discretization step in Angstrom. Will be 
                              adjusted due to rounding.
        @param x_eval_rgion   Evaluation region in Angstrom. If set, no pbc 
                              are applied.
        @param eval_cutoff    Cutoff for orbital evaluation in Angstrom.
        @param reserve_extrap Workfunction in eV (I pulled a sneaky on you!).
        @param print_info     Print calculation information.
        """

        time1 = time.time()

        dr_guess *= ang_2_bohr
        eval_cutoff *= ang_2_bohr
        kappa = np.sqrt(2.0 / hart_2_ev * reserve_extrap)

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

        ext_z_n = 0 #int(np.round(reserve_extrap/self.dv[2]))

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
            r_vec = r_vec_2**0.5

            for ispin in range(nspin):
                morb_grids_local[ispin].fill(0.0)

            for i_set, bset in enumerate(self.basis_sets[elem]):
                for i_shell, shell in enumerate(bset):
                    l = shell[0]
                    if l > 1:
                        continue

                    # Calculate the radial part of the atomic orbital
                    time2 = time.time()
                    radial_part = np.exp(-kappa*r_vec)
                    time_radial_calc += time.time() - time2

                    for i_orb, m in enumerate(range(-l, l+1, 1)):
                        time2 = time.time()
                        atomic_orb = radial_part*self._spherical_harmonic_grid(l, m,
                                                                        rel_loc_cell_grids[0],
                                                                        rel_loc_cell_grids[1],
                                                                        rel_loc_cell_grids[2], r_vec)
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
                    self._add_local_to_global_grid(
                            morb_grids_local[i_spin][i_mo],
                            self.morb_grids[i_spin][i_mo],
                            origin_diff,
                            wrap=(mid_ixs != -1))
            time_loc_glob_add += time.time() - time2

        if print_info:
            print("---- Radial calc time : %4f" % time_radial_calc)
            print("---- Spherical calc time : %4f" % time_spherical)
            print("---- Loc -> loc_morb time : %4f" % time_loc_lmorb_add)
            print("---- loc_morb -> glob time : %4f" % time_loc_glob_add)
            print("---- Total time: %.4f"%(time.time() - time1))


    ### ------------------------------------------------------------------------
    ### Extrapolate wavefunctions
    ### ------------------------------------------------------------------------

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
        pass

    def extrapolate_morbs_spin(self, ispin, vacuum_pot=None, hart_plane=None, use_weighted_avg=True):
        pass
