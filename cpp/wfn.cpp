// STL includes ----------------------------------------------------------------
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
// OpenMP includes -------------------------------------------------------------
#include <omp.h>
// Python includes -------------------------------------------------------------
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// Own includes ----------------------------------------------------------------
#include "wfn.hpp"


extern "C" {

  /*! @brief Evaluates the wavefunction and its derivatives on a grid.
   *
   *  @param wn           Value for workfunction.
   *  @param rCut         Cutoff radius for sample wavefunction.
   *  @param pbc          Array with booleans for periodicity.
   *  @param abc          Size of box used for atoms and periodicity.
   *  @param coeffsObject Basis coefficients.
   *  @param atoms        Positions of atoms.
   *  @param grid         Grid points.
   *  @param wfnObject    Output structure assumed to be set zero.
   */
  void computeWFN( const double wn,
                   const double rCut,
                   const bool *pbc,
                   const double *abc, 
                   PyArrayObject *coeffsObject,
                   PyArrayObject *atoms,
                   PyArrayObject *grid,
                   PyArrayObject *wfnObject ) {
    // Decay
    const double kappa = std::sqrt(2.0*EV2HARTREE*wn);
    // Extra information from containers
    const std::size_t noDer = PyArray_DIMS(wfnObject)[1];
    const std::size_t noCoeffs = PyArray_DIMS(coeffsObject)[2];
    const std::size_t noEigs = PyArray_DIMS(coeffsObject)[1];
    const std::size_t noPoints = PyArray_DIMS(grid)[0];
    const std::size_t noAtoms = PyArray_DIMS(atoms)[0];
    // Converting to useful containers
    const Vector3d* R = (const Vector3d*) PyArray_DATA(grid);
    const Atoms A((const Vector3d*) PyArray_DATA(atoms), noAtoms, rCut, pbc, 
      *(const Vector3d*) abc);
    const double *coeffs = (const double *) PyArray_DATA(coeffsObject);
    double *wfn = (double *) PyArray_DATA(wfnObject);

    // Loop over grid points and compute wavefunctions
    #pragma omp parallel for
    for( std::size_t pointIdx = 0; pointIdx < noPoints; ++pointIdx ) {
      const std::vector<double> tmp = atomLoop( kappa, R[pointIdx], noAtoms, 
        noCoeffs, noEigs, noDer, A, coeffs );
      for( std::size_t derIdx = 0; derIdx < noDer; ++derIdx )
        for( std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx )
          wfn[eigIdx*noDer*noPoints+derIdx*noPoints+pointIdx] += 
            tmp[derIdx*noEigs+eigIdx];
    }
  } // end computeWFN

} // end extern "C"
