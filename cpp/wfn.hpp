#ifndef WFN_HPP
#define WFN_HPP
// STL includes ----------------------------------------------------------------
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>
// Own includes ----------------------------------------------------------------
#include "atomCL.hpp"
#include "basis.hpp"
#include "vector.hpp"


/*! @brief Evaluates wavefunctions at specific grid point.
 *
 *  Wavefunctions here refers to the wavefunction and a number of differential
 *  operators applied to it. The wavefunction itself is computed using the basis
 *  from the PPSTM code. The differential operators correspond to the tip 
 *  orbitals where:
 *    ( s, py, pz, px ) <-> ( I, d/(kappa*dy), d/(kappa*dz), d/(kappa*dx) )
 *
 *  @param kappa    Decay constant.
 *  @param r        Grid point.
 *  @param noAtoms  Number of atoms.
 *  @param noCoeffs Number of coefficients per atomic orbital.
 *  @param noEigs   Number of eigenvalues.
 *  @param noDer    Number of differential operators evaluated.
 *  @param A        Atom object.
 *  @param coeffs   C-array with basis function coefficients.
 *
 *  @return A vector with wavefunctions values.
 */
inline std::vector<double> atomLoop( const double wn,
                                     const Vector3d &r,
                                     const std::size_t noAtoms,
                                     const std::size_t noCoeffs,
                                     const std::size_t noEigs,
                                     const std::size_t noDer,
                                     const Atoms &A,
                                     const double *coeffs,
                                     const double *eigs ) {
  // Result
  std::vector<double> res(noDer*noEigs, 0.0);
  double *resData = res.data();

  const std::size_t noTotCoeffs = noCoeffs*noEigs;
  const auto func = [&noDer, &noCoeffs, &wn, &noEigs, &coeffs, &eigs, &noTotCoeffs, &resData] (Vector3d dr, std::size_t idx) {
    dr *= ANG2BOHR;
    const double nr = dr.norm();
    const double nr_inv = 1. / nr;
    const double* coeffsPointer = coeffs+(noTotCoeffs*idx);
    /* s-tip, s-sample */
    if( noCoeffs == 1 && noDer == 1 )
      ss(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
    /* s-tip, p-sample */
    else if( noDer == 1 && noCoeffs == 4 )
      ssp(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
    /* s-tip, d-sample */
    else if( noDer == 1 && noCoeffs == 9 )
      sspd(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
    /* p-tip, s-sample */
    else if( noDer == 4 && noCoeffs == 1 ) {
      ss(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
      pys(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs);
      pzs(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*2);
      pxs(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*3);
    }
    /* p-tip, p-sample */
    else if( noDer == 4 && noCoeffs == 4 ) {
      ssp(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
      pysp(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs);
      pzsp(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*2);
      pxsp(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*3);
    }
    /* p-tip, d-sample */
    else if( noDer == 4 && noCoeffs == 9 ) {
      sspd(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData);
      pyspd(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs);
      pzspd(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*2);
      pxspd(wn, noEigs, coeffsPointer, eigs, nr_inv, nr, dr, resData+noEigs*3);
    }
    else {
      throw std::invalid_argument("Tip-Sample combination is not supported.\n");
    }  
  };

  A.doFor(r, func);

  return res;
} // end atomLoop

#endif // WFN_HPP
