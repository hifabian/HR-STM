// STL includes ----------------------------------------------------------------
#include <algorithm> // copy
#include <array>     // array
#include <cmath>     // exp, sqrt, pow, copysign
#include <cstddef>   // size_t
#include <map>       // map
#include <utility>   // pair, make_pair
#include <vector>    // vector
// OpenMP includes -------------------------------------------------------------
#include <omp.h>
// Python includes -------------------------------------------------------------
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// Own includes ----------------------------------------------------------------
#include "basis.hpp"
#include "vector.hpp"
#include "wfn.hpp"


/*! @brief Gaussian density of states.
 *
 *  @param val   Input value.
 *  @param sigma Variance.
 *
 *  @return Gaussian evaluated at input value.
 */
inline double dosGauss( const double val, const double sigma ) {
  return std::exp( -val*val/(sigma*sigma) ) / (sigma*std::sqrt(2*M_PI));
}

/*! @brief Lorentzian density of states.
 *
 *  @param val   Input value.
 *  @param eta   Half of the broadening factor.
 *
 *  @return Lorentzian evaluated at input value.
 */
inline double dosLorentz( const double val, const double eta ) {
  return -1./M_PI * eta / ( val*val + eta*eta );
}

extern "C" {

  /*! @brief Computes the current on a grid.
   *
   *  The basis functions are the same as in the PPSTM code.
   *  The density of states is taken as a Gaussian (for now).
   *
   *  @attention Generally, methods without additional checks are used
   *             by this function. Thus the input is assumed to be proper.
   *
   *  @note The PPSTM code uses a different criterion for whether current can flow
   *        or not. Here we check how the eigenenergies are relative to the Fermi 
   *        level (which is 0.0eV) thus the broadening has no impact on which
   *        states allow tunnelling. In the PPSTM code, on the other hand, the 
   *        density of states is cut at the Fermi level meaning that broadening
   *        can have a significant impact on tunnelling close to 0.0eV bias voltage.
   *
   *  @param wn         Workfunction in eV.
   *  @param sigma      Variance for density of states.
   *  @param rot        Flag for rotating coefficents.
   *  @param rCut       Cutoff radius for sample wavefunction.
   *  @param pbc        Array with booleans for periodicity.
   *  @param abc        Size of box used for atoms and periodicity.
   *  @param voltages   Voltages.
   *  @param coeffsTip  Coefficients of derivative rule (untransformed).
   *  @param coeffsSam  Coefficients for wavefunction on sample.
   *  @param eigsTip    Eigenvalues on tip.
   *  @param eigsSam    Eigenvalues on sample.
   *  @param grids      Grids on which current is evaluated (i.e. apex-atom
                        positions to which tunnelling occurs). Proceeding grids
                        are taken as reference for displacement using subtraction.
                        The first grid is a reference with no tunnelling to it.
   *  @param atoms      Atom positions of sample. Used to compute wavefunction.
   *  @param current    Output structure assumed to be set zero.
   */
  void computeCurrent( const double wn, const double sigma, const bool rot,
                       const double rCut, const bool *pbc,
                       const double *abc,
                       PyArrayObject *voltages,
                       PyObject *coeffsTip, PyObject *coeffsSam,
                       PyObject *eigsTip,   PyObject *eigsSam,
                       PyObject *grids,     PyArrayObject *atoms,
                       PyArrayObject *current ) { 
    // Extra information from containers
    const std::size_t noDer = PyArray_DIMS((PyArrayObject*) PyList_GET_ITEM(PyList_GET_ITEM(coeffsTip,0),0))[1];
    const std::size_t noCoeffs = PyArray_DIMS((PyArrayObject*) PyList_GET_ITEM(coeffsSam,0))[2];
    const std::size_t noSpinsTip = PyList_GET_SIZE(PyList_GET_ITEM(coeffsTip,0));
    const std::size_t noSpinsSam = PyList_GET_SIZE(coeffsSam);
    const std::size_t noEigsTip = PyArray_DIMS((PyArrayObject*) PyList_GET_ITEM(eigsTip,0))[0];
    const std::size_t noTunnels = PyList_GET_SIZE(coeffsTip);
    const std::size_t noGrids  = PyList_GET_SIZE(grids);
    const std::size_t noPoints = PyArray_DIMS((PyArrayObject*) PyList_GET_ITEM(grids,0))[0];
    const std::size_t noAtoms = PyArray_DIMS(atoms)[0];
    const std::size_t noVoltages = PyArray_DIMS(voltages)[0];
    // Atoms object
    const Atoms A((const Vector3d*) PyArray_DATA(atoms), noAtoms, rCut, pbc, *(const Vector3d*) abc);

    // Determine all potentially relevant eigTip-eigSam pairs.
    // If you want PPSTM pairs, check for this instead where appropriate:
    //   (eigTip <= 0 && eigTip+voltage > 0) || (eigTip > 0 && eigTip+voltage <= 0)
    typedef std::pair<std::size_t, std::size_t> IndPair;
    std::map<IndPair, std::vector<IndPair>> e2e;
    for( std::size_t spinSamIdx = 0; spinSamIdx < noSpinsSam; ++spinSamIdx ) {
      PyArrayObject *eigsSamObject = (PyArrayObject*) PyList_GET_ITEM(eigsSam,spinSamIdx);
      PyArrayObject *coeffsSamObject = (PyArrayObject*) PyList_GET_ITEM(coeffsSam,spinSamIdx);
      const std::size_t noEigsSam = PyArray_DIMS(eigsSamObject)[0];
      for( std::size_t eigSamIdx = 0; eigSamIdx < noEigsSam; ++eigSamIdx ) {
        const double eigSam = *(const double*) PyArray_GETPTR1(eigsSamObject,eigSamIdx);
        for( std::size_t spinTipIdx = 0; spinTipIdx < noSpinsTip; ++spinTipIdx ) {
          PyArrayObject *eigsTipObject = (PyArrayObject*) PyList_GET_ITEM(eigsTip,spinTipIdx);
          const std::size_t noEigsTip = PyArray_DIMS(eigsTipObject)[0];
          for( std::size_t eigTipIdx = 0; eigTipIdx < noEigsTip; ++eigTipIdx ) {
            const double eigTip = *(const double*) PyArray_GETPTR1(eigsTipObject,eigTipIdx);
            if( eigTip*eigSam > 0.0 ||
              (eigTip == 0.0 && eigSam <= 0.0) ||
              (eigSam == 0.0 && eigTip <= 0.0) )
              continue; // Both are either occupied or unoccupied
            for( std::size_t volIdx = 0; volIdx < noVoltages; ++volIdx ) {
              const double voltage = *(const double *) PyArray_GETPTR1(voltages,volIdx);
              // 4.8916384757 for 99.9999%, 3.8905918864 for 99.99%
              if( std::abs(voltage+eigTip-eigSam) < 4.0*sigma ) {
                e2e[std::make_pair(spinSamIdx,eigSamIdx)].push_back(std::make_pair(spinTipIdx,eigTipIdx));
                break;
              }
            } // end for volIdx
          } // end for eigTipIdx
        } // end for spinTipIdx
      } // end for eigSamIdx
    } // end for spinSamIdx

    #pragma omp parallel for
    for( std::size_t pointIdx = 0; pointIdx < noPoints; ++pointIdx ) {
      for( std::size_t tunnelIdx = 0; tunnelIdx < noTunnels; ++tunnelIdx ) {
        // Get grid point (converting: PyObject -> PyArrayObject* -> void* -> double* -> Vector3d&)
        const Vector3d &curGrid = *(const Vector3d*) (const double*) 
          PyArray_GETPTR1((PyArrayObject*) PyList_GET_ITEM(grids,tunnelIdx+1),pointIdx);
        const Vector3d &preGrid = *(const Vector3d*) (const double*) 
          PyArray_GETPTR1((PyArrayObject*) PyList_GET_ITEM(grids,tunnelIdx),pointIdx);

        // Rotate coefficients
        std::array<double, 9> rotMatrix = {1.0,0.0,0.0,
                                           0.0,1.0,0.0,
                                           0.0,0.0,1.0};
        if( rot && noDer == 4 ) {
          const Vector3d v = {0.0,0.0,-1.0};
          Vector3d w = curGrid-preGrid;
          w /= w.norm();
          Vector3d n = cross(v,w);
          if( n.norm() > 1e-12 ) { // Not rotated
            n /= n.norm();
            const double cosa = v*w;
            const double sina = std::sqrt(1.0-cosa*cosa);
            rotMatrix = {n[1]*n[1]*(1-cosa)+cosa, n[2]*n[1]*(1-cosa)-n[0]*sina, n[0]*n[1]*(1-cosa)+n[2]*sina,
                         n[1]*n[2]*(1-cosa)+n[0]*sina, n[2]*n[2]*(1-cosa)+cosa, n[0]*n[2]*(1-cosa)-n[1]*sina,
                         n[1]*n[0]*(1-cosa)-n[2]*sina, n[2]*n[0]*(1-cosa)+n[1]*sina, n[0]*n[0]*(1-cosa)+cosa};
          }
        }
          
        for( std::size_t spinSamIdx = 0; spinSamIdx < noSpinsSam; ++spinSamIdx ) {
          PyArrayObject *eigsSamObject = (PyArrayObject*) PyList_GET_ITEM(eigsSam,spinSamIdx);
          PyArrayObject *coeffsSamObject = (PyArrayObject*) PyList_GET_ITEM(coeffsSam,spinSamIdx);
          const std::size_t noEigsSam = PyArray_DIMS(eigsSamObject)[0];
          // Wavefunctions and its derivatives for all eigenvalues on sample
          const auto wfn = atomLoop( wn, curGrid, noAtoms, noCoeffs, noEigsSam, noDer, A, 
            (const double*) PyArray_DATA(coeffsSamObject),
            (const double*) PyArray_DATA(eigsSamObject));

          for( std::size_t eigSamIdx = 0; eigSamIdx < noEigsSam; ++eigSamIdx ) {
            const double eigSam = *(const double*) PyArray_GETPTR1(eigsSamObject,eigSamIdx);

            // No corresponding energies (this allows using at()-operator,
            // allows for sharing e2e among threads)
            if( e2e.count(std::make_pair(spinSamIdx,eigSamIdx)) == 0 )
              continue;

            for( const auto &ip : e2e.at(std::make_pair(spinSamIdx,eigSamIdx)) ) {
              const std::size_t spinTipIdx = ip.first;
              const std::size_t eigTipIdx = ip.second;

              PyArrayObject *eigsTipObject = (PyArrayObject*) PyList_GET_ITEM(eigsTip,spinTipIdx);
              PyArrayObject *coeffsTipObject = (PyArrayObject*) PyList_GET_ITEM(
                PyList_GET_ITEM(coeffsTip,tunnelIdx),spinTipIdx);
              const std::size_t noEigsTip = PyArray_DIMS(eigsTipObject)[0];

              const double eigTip = *(const double*) PyArray_GETPTR1(eigsTipObject,eigTipIdx);
              double *chenCoeffsUn = (double*) PyArray_GETPTR1(coeffsTipObject,eigTipIdx);
              double chenCoeffs[noDer];

              chenCoeffs[0] = chenCoeffsUn[0];
              if( noDer > 1 ) {
                chenCoeffs[1] = rotMatrix[0]*chenCoeffsUn[1] 
                    + rotMatrix[1]*chenCoeffsUn[2]
                    + rotMatrix[2]*chenCoeffsUn[3];
                  chenCoeffs[2] = rotMatrix[3]*chenCoeffsUn[1]
                    + rotMatrix[4]*chenCoeffsUn[2]
                    + rotMatrix[5]*chenCoeffsUn[3];
                  chenCoeffs[3] = rotMatrix[6]*chenCoeffsUn[1]
                    + rotMatrix[7]*chenCoeffsUn[2]
                    + rotMatrix[8]*chenCoeffsUn[3];
              }

              // Tunneling matrix
              double tunnelMatrix = 0.0;
              for( std::size_t derIdx = 0; derIdx < noDer; ++derIdx )
                tunnelMatrix += chenCoeffs[derIdx]*wfn[derIdx*noEigsSam+eigSamIdx];

              for( std::size_t volIdx = 0; volIdx < noVoltages; ++volIdx ) {
                const double voltage = *(const double*) PyArray_GETPTR1(voltages,volIdx);
                // 4.8916384757 for 99.9999%, 3.8905918864 for 99.99%
                if( std::abs(voltage+eigTip-eigSam) >= 4.0*sigma )
                  continue;
                double &cur = *(double*) PyArray_GETPTR2(current,pointIdx,volIdx);
                cur += std::copysign(1.0,eigTip)*dosGauss(voltage+eigTip-eigSam,sigma)*std::pow(tunnelMatrix,2);
              } // end for volIdx
            } // end for ip
          } // end for eigSamIdx
        } // end for spinSamIdx
      } // end for tunnelIdx
    } // end for pointIdx
  } // end computeCurrent

} // end extern "C"
