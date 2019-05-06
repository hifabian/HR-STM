#ifndef BASIS_HPP
#define BASIS_HPP
// STL includes ----------------------------------------------------------------
#include <cstddef>
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

inline void ss(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx]*radial;      // s-orbital on sample
  }
}
inline void pys(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[1]*kappa;
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx]*sOrb; // s-orbtial on sample
  }
}
inline void pzs(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[2]*kappa;
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx]*sOrb; // s-orbtial on sample
  }
}
inline void pxs(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[0]*kappa;
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx]*sOrb; // s-orbtial on sample
  }
}


// =============================================================================
//                                 sp-Sample
// =============================================================================

inline void ssp(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double pyOrb = SQRT_3*radial*nr_inv*dr[1];
  const double pzOrb = SQRT_3*radial*nr_inv*dr[2];
  const double pxOrb = SQRT_3*radial*nr_inv*dr[0];
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*4]*radial;  //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*pyOrb; // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*pzOrb; // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*pxOrb; // px-orbital on sample
  }
}
inline void pysp(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[1]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[1]*dr[1]*(kappa+nr_inv)); 
  const double pzOrb = SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[1]*(kappa+nr_inv);
  const double pxOrb = SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[1]*(kappa+nr_inv);
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*4]*sOrb;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*pyOrb; // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*pzOrb; // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*pxOrb; // px-orbital on sample
  }
}
inline void pzsp(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[2]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[2]*(kappa+nr_inv);
  const double pzOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[2]*dr[2]*(kappa+nr_inv)); 
  const double pxOrb = SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[2]*(kappa+nr_inv);
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*4]*sOrb;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*pyOrb; // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*pzOrb; // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*pxOrb; // px-orbital on sample
  }
}
inline void pxsp(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[0]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[0]*(kappa+nr_inv);
  const double pzOrb = SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[0]*(kappa+nr_inv);
  const double pxOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[0]*dr[0]*(kappa+nr_inv)); 
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*4]*sOrb;    //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+1]*pyOrb; // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+2]*pzOrb; // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*4+3]*pxOrb; // px-orbital on sample
  }
}

// =============================================================================
//                                 spd-Sample
// =============================================================================

inline void sspd(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double pyOrb = SQRT_3*radial*nr_inv*dr[1];
  const double pzOrb = SQRT_3*radial*nr_inv*dr[2];
  const double pxOrb = SQRT_3*radial*nr_inv*dr[0];
  const double dxyOrb = SQRT_15*radial*nr_inv*nr_inv*dr[0]*dr[1];
  const double dyzOrb = SQRT_15*radial*nr_inv*nr_inv*dr[1]*dr[2];
  const double dz2Orb = SQRT_5*0.5*radial*(3.0*nr_inv*nr_inv*dr[2]*dr[2] - 1.0);
  const double dxzOrb = SQRT_15*radial*nr_inv*nr_inv*dr[2]*dr[0];
  const double dx2y2Orb = SQRT_15*0.5*radial*nr_inv*nr_inv*(dr[0]*dr[0]-dr[1]*dr[1]);
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*9]*radial;     //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*pyOrb;    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*pzOrb;    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*pxOrb;    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*dxyOrb;   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*dyzOrb;   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*dz2Orb;   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*dxzOrb;   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*dx2y2Orb; // dx2-y2-orbital on sample
  }
}
inline void pyspd(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[1]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[1]*dr[1]*(kappa+nr_inv)); 
  const double pzOrb = SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[1]*(kappa+nr_inv);
  const double pxOrb = SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[1]*(kappa+nr_inv);
  const double dxyOrb = SQRT_15*radial*nr_inv*dr[0]*(-1.0+2.0*nr_inv*dr[1]*dr[1]*(kappa+nr_inv));
  const double dyzOrb = SQRT_15*radial*nr_inv*dr[2]*(-1.0+2.0*nr_inv*dr[1]*dr[1]*(kappa+nr_inv));
  const double dz2Orb = SQRT_5*0.5*radial*dr[1]*(3.0*dr[2]*dr[2]*nr_inv*nr_inv*(kappa+2.0*nr_inv)-kappa);
  const double dxzOrb = SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);
  const double dx2y2Orb = SQRT_15*radial*nr_inv*dr[1]*(1.0+nr_inv*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa));
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*9]*sOrb;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*pyOrb;    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*pzOrb;    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*pxOrb;    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*dxyOrb;   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*dyzOrb;   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*dz2Orb;   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*dxzOrb;   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*dx2y2Orb; // dx2-y2-orbital on sample
  }
}
inline void pzspd(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[2]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[2]*(kappa+nr_inv);
  const double pzOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[2]*dr[2]*(kappa+nr_inv)); 
  const double pxOrb = SQRT_3*radial*nr_inv*nr_inv*dr[0]*dr[2]*(kappa+nr_inv);
  const double dxyOrb = SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);
  const double dyzOrb = SQRT_15*radial*nr_inv*dr[1]*(-1.0+2.0*nr_inv*dr[2]*dr[2]*(kappa+nr_inv));
  const double dz2Orb = SQRT_5*0.5*radial*(-kappa+3.0*nr_inv*dr[2]*(-2.0+nr_inv*dr[2]*(kappa+2.0*nr_inv*dr[2])));
  const double dxzOrb = SQRT_15*radial*nr_inv*dr[0]*(-1.0+2.0*nr_inv*dr[2]*dr[2]*(kappa+nr_inv));
  const double dx2y2Orb = SQRT_15*radial*nr_inv*nr_inv*dr[2]*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa);
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*9]*sOrb;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*pyOrb;    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*pzOrb;    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*pxOrb;    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*dxyOrb;   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*dyzOrb;   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*dz2Orb;   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*dxzOrb;   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*dx2y2Orb; // dx2-y2-orbital on sample
  }
}
inline void pxspd(const double kappa, const std::size_t noEigs, const double* coeffs, 
               const double nr_inv, const Vector3d& dr, const double radial,
               double* res) {
  const double sOrb = radial*nr_inv*dr[0]*kappa;
  const double pyOrb = SQRT_3*radial*nr_inv*nr_inv*dr[1]*dr[0]*(kappa+nr_inv);
  const double pzOrb = SQRT_3*radial*nr_inv*nr_inv*dr[2]*dr[0]*(kappa+nr_inv);
  const double pxOrb = SQRT_3*radial*nr_inv*(-1.0+nr_inv*dr[0]*dr[0]*(kappa+nr_inv)); 
  const double dxyOrb = SQRT_15*radial*nr_inv*dr[1]*(-1.0+2.0*nr_inv*dr[0]*dr[0]*(kappa+nr_inv));
  const double dyzOrb = SQRT_15*radial*dr[0]*dr[1]*dr[2]*nr_inv*nr_inv*(2.0*nr_inv+kappa);
  const double dz2Orb = SQRT_5*0.5*radial*dr[0]*(3.0*dr[2]*dr[2]*nr_inv*nr_inv*(kappa+2.0*nr_inv)-kappa);
  const double dxzOrb = SQRT_15*radial*nr_inv*dr[2]*(-1.0+2.0*nr_inv*dr[0]*dr[0]*(kappa+nr_inv));
  const double dx2y2Orb = SQRT_15*radial*nr_inv*dr[0]*(-1.0+nr_inv*(dr[0]*dr[0]-dr[1]*dr[1])*(nr_inv+0.5*kappa));
  for(std::size_t eigIdx = 0; eigIdx < noEigs; ++eigIdx) {
    res[eigIdx] += coeffs[eigIdx*9]*sOrb;       //  s-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+1]*pyOrb;    // py-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+2]*pzOrb;    // pz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+3]*pxOrb;    // px-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+4]*dxyOrb;   // dxy-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+5]*dyzOrb;   // dyz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+6]*dz2Orb;   // dz2-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+7]*dxzOrb;   // dxz-orbital on sample
    res[eigIdx] += coeffs[eigIdx*9+8]*dx2y2Orb; // dx2-y2-orbital on sample
  }
}
// =============================================================================
// =============================================================================

#endif // BASIS_HPP
