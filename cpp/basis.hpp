#ifndef BASIS_HPP
#define BASIS_HPP
// STL includes ----------------------------------------------------------------
#include <cstddef>
#include <cmath>
// Own includes ----------------------------------------------------------------
#include "vector.hpp"

// =============================================================================
// =============================================================================

// Useful constants
#define SQRT_3     1.73205080757
#define SQRT_15    3.87298334621
#define SQRT_5     2.23606797750
#define ANG2BOHR   1.88972612546
#define EV2HARTREE 0.03674930814

// =============================================================================
// =============================================================================


// =============================================================================
//                                 s-Sample
// =============================================================================

inline void ss(const double wn, const std::size_t noEigs, const double* coeffs, 
               const double* eigs, const double nr_inv, const double nr,
               const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx]*radial;      // s-orbital on sample
  }
}
inline void pys(const double wn, const std::size_t noEigs, const double* coeffs, 
                const double* eigs, const double nr_inv, const double nr,
                const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx]*radial*nr_inv*dr[1]*kappa; // s-orbtial on sample
  }
}
inline void pzs(const double wn, const std::size_t noEigs, const double* coeffs, 
                const double* eigs, const double nr_inv, const double nr,
                const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx]*radial*nr_inv*dr[2]*kappa; // s-orbtial on sample
  }
}
inline void pxs(const double wn, const std::size_t noEigs, const double* coeffs, 
                const double* eigs, const double nr_inv, const double nr,
                const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx]*radial*nr_inv*dr[0]*kappa; // s-orbtial on sample
  }
}


// =============================================================================
//                                 sp-Sample
// =============================================================================

inline void ssp(const double wn, const std::size_t noEigs, const double* coeffs, 
                const double* eigs, const double nr_inv, const double nr,
                const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*4]*radial;  //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*SQRT_3*radial*nr_inv*dr[1]; // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*SQRT_3*radial*nr_inv*dr[2]; // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*SQRT_3*radial*nr_inv*dr[0]; // px-orbital on sample
  }
}
inline void pysp(const double wn, const std::size_t noEigs, const double* coeffs, 
                 const double* eigs, const double nr_inv, const double nr,
                 const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*4]*radial*nr_inv*dr[1]*kappa;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[1]*dr[1]*(kappa+nr_inv)); // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[1]*(kappa+nr_inv); // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[1]*(kappa+nr_inv); // px-orbital on sample
  }
}
inline void pzsp(const double wn, const std::size_t noEigs, const double* coeffs, 
                 const double* eigs, const double nr_inv, const double nr,
                 const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*4]*radial*nr_inv*dr[2]*kappa;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[2]*(kappa+nr_inv); // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[2]*dr[2]*(kappa+nr_inv)); // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[2]*(kappa+nr_inv); // px-orbital on sample
  }
}
inline void pxsp(const double wn, const std::size_t noEigs, const double* coeffs, 
                 const double* eigs, const double nr_inv, const double nr,
                 const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*4]*radial*nr_inv*dr[0]*kappa;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[0]*(kappa+nr_inv); // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[0]*(kappa+nr_inv); // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[0]*dr[0]*(kappa+nr_inv)); // px-orbital on sample
  }
}

// =============================================================================
//                                 spd-Sample
// =============================================================================

inline void sspd(const double wn, const std::size_t noEigs, const double* coeffs, 
                 const double* eigs, const double nr_inv, const double nr,
                 const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*9]*radial;     //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*SQRT_3*radial*nr_inv*dr[1];    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*SQRT_3*radial*nr_inv*dr[2];    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*SQRT_3*radial*nr_inv*dr[0];    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*SQRT_15*radial*nr_inv*nr_inv*dr[0]*dr[1];   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*SQRT_15*radial*nr_inv*nr_inv*dr[1]*dr[2];   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*SQRT_5*0.5*radial*(3.0*nr_inv*nr_inv*dr[2]*dr[2] - 1.0);   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*SQRT_15*radial*nr_inv*nr_inv*dr[2]*dr[0];   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*SQRT_15*0.5*radial*nr_inv*nr_inv*(dr[0]*dr[0]-dr[1]*dr[1]); // dx2-y2-orbital on sample
  }
}
inline void pyspd(const double wn, const std::size_t noEigs, const double* coeffs, 
                  const double* eigs, const double nr_inv, const double nr,
                  const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*9]*radial*nr_inv*dr[1]*kappa;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[1]*dr[1]*(kappa+nr_inv));    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[1]*(kappa+nr_inv);    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[1]*(kappa+nr_inv);    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*SQRT_15*radial*nr_inv*dr[0]*(-1.0+2.0*nr_inv*dr[1]*dr[1]*(kappa+nr_inv));   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*SQRT_15*radial*nr_inv*dr[2]*(-1.0+2.0*nr_inv*dr[1]*dr[1]*(kappa+nr_inv));   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*SQRT_5*0.5*radial*dr[1]*(3.0*dr[2]*dr[2]*nr_inv*nr_inv*(kappa+2.0*nr_inv)-kappa);   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*SQRT_15*radial*nr_inv*dr[1]*(1.0+nr_inv*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa)); // dx2-y2-orbital on sample
  }
}
inline void pzspd(const double wn, const std::size_t noEigs, const double* coeffs, 
                  const double* eigs, const double nr_inv, const double nr,
                  const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*9]*radial*nr_inv*dr[2]*kappa;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[2]*(kappa+nr_inv);    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[2]*dr[2]*(kappa+nr_inv));    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[2]*(kappa+nr_inv);    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*SQRT_15*radial*nr_inv*dr[1]*(-1.0+2.0*nr_inv*dr[2]*dr[2]*(kappa+nr_inv));   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*SQRT_5*0.5*radial*(-kappa+3.0*nr_inv*dr[2]*(-2.0+nr_inv*dr[2]*(kappa+2.0*nr_inv*dr[2])));   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*SQRT_15*radial*nr_inv*dr[0]*(-1.0+2.0*nr_inv*dr[2]*dr[2]*(kappa+nr_inv));   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*SQRT_15*radial*nr_inv*nr_inv*dr[2]*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa); // dx2-y2-orbital on sample
  }
}
inline void pxspd(const double wn, const std::size_t noEigs, const double* coeffs, 
                  const double* eigs, const double nr_inv, const double nr,
                  const Vector3d& dr, double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    const double kappa = std::sqrt(2.0*EV2HARTREE*(wn-eigs[eigIdx])); // Decay constant
    const double radial = std::exp(-nr*kappa);             // Radial part
    res[eigIdx] += coeffs[eigIdx*9]*radial*nr_inv*dr[0]*kappa;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[0]*(kappa+nr_inv);    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[0]*(kappa+nr_inv);    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[0]*dr[0]*(kappa+nr_inv));    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*SQRT_15*radial*nr_inv*dr[1]*(-1.0+2.0*nr_inv*dr[0]*dr[0]*(kappa+nr_inv));   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*SQRT_5*0.5*radial*dr[0]*(3.0*dr[2]*dr[2]*nr_inv*nr_inv*(kappa+2.0*nr_inv)-kappa);   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*SQRT_15*radial*nr_inv*dr[2]*(-1.0+2.0*nr_inv*dr[0]*dr[0]*(kappa+nr_inv));   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*SQRT_15*radial*nr_inv*dr[0]*(-1.0+nr_inv*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa)); // dx2-y2-orbital on sample
  }
}
// =============================================================================
// =============================================================================

#endif // BASIS_HPP
