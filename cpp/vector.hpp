#ifndef VECTOR_HPP
#define VECTOR_HPP
// STL includes ----------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
// Boost includes --------------------------------------------------------------
//#include <boost/mpl/range_c.hpp>
//#include <boost/mpl/for_each.hpp>


/*! @brief Evaluates a function f(i) as i goes from 0 to n.
 *
 *  @tparam n End of loop (exclusive).
 *  @param f Function.
 */
template< std::size_t n, class F >
inline void loop( F f ) { 
  for( std::size_t i = 0; i < n; ++i )
    f(i);
//  return boost::mpl::for_each< boost::mpl::range_c<std::size_t,0,n> >(f); 
}


/*! @brief Vector class.
 *
 *  Provides several functionalities useful for vectors of arbitrary size.
 *  Not all functions make sense and depend on the scalar field.
 *
 *  @tparam SCALAR Scalar field.
 *  @tparam SIZE   Size of vector.
 */
template< typename SCALAR, std::size_t SIZE >
class Vector {
public:
  //! Renaming scalar field.
  using scalar_t = SCALAR;
  //! Convenient typedef for this Vector type.
  using self_t = Vector<SCALAR, SIZE>;

private:
  //! Internal storage as C-array.
  scalar_t data[SIZE] = {0};

public:
  //! Default constructor.
  Vector() {}
  //! Constructor using array.
  Vector( const scalar_t data[SIZE] ) {
    loop<SIZE>( [this,data](std::size_t i){ this->data[i] = data[i]; });
  }
  //! Constructor using initializer list.
  Vector( const std::initializer_list<scalar_t> data ) {
    std::copy(std::begin(data), std::end(data), std::begin(this->data));
  }


  //! Bracket operator.
  inline scalar_t& operator[](std::size_t i) {
    return data[i];
  }
  //! Const bracket oprator.
  inline const scalar_t& operator[](std::size_t i) const {
    return data[i];
  }

  //! Equal operator.
  inline self_t& operator=( const self_t& a ) {
    loop<SIZE>([this,&a](std::size_t i){ this->data[i] = a[i]; });
  } 
  //! Plus-equal operator.
  inline self_t& operator+=( const Vector<SCALAR,SIZE>& a ) {
    loop<SIZE>([this,&a](std::size_t i){ this->data[i] += a[i]; });
  }
  //! Minus-equal operator.
  inline self_t& operator-=( const Vector<SCALAR,SIZE>& a ) {
    loop<SIZE>([this,&a](std::size_t i){ this->data[i] -= a[i]; });
  }
  //! Scalar-multplication-equal operator.
  inline self_t& operator*=( const SCALAR& c ) {
    loop<SIZE>([this,&c](std::size_t i){ this->data[i] *= c; });
  }
  //! Scalar-divison-equal operator.
  inline self_t& operator/=( const SCALAR& c ) {
    loop<SIZE>([this,&c](std::size_t i){ this->data[i] /= c; });
  }

  //! Norm of vector.
  inline scalar_t norm() const {
    scalar_t res(0);
    loop<SIZE>([this,&res](std::size_t i){ res += this->data[i]*this->data[i]; });
    return std::sqrt(res);
  }
  inline scalar_t norm2() const {
    scalar_t res(0);
    loop<SIZE>([this,&res](std::size_t i){ res += this->data[i]*this->data[i]; });
    return res;
  }
};

//! Addition operator.
template< typename SCALAR, std::size_t SIZE >
inline Vector<SCALAR,SIZE> operator+(const Vector<SCALAR,SIZE>& a, const Vector<SCALAR,SIZE>& b) {
  Vector<SCALAR,SIZE> c = a;
  c += b;
  return c;
}
//! Subtraction operator.
template< typename SCALAR, std::size_t SIZE >
inline Vector<SCALAR,SIZE> operator-(const Vector<SCALAR,SIZE>& a, const Vector<SCALAR,SIZE>& b) {
  Vector<SCALAR,SIZE> c = a;
  c -= b;
  return c;
}
//! Inner-product.
template< typename SCALAR, std::size_t SIZE >
inline SCALAR operator*(const Vector<SCALAR,SIZE>& a, const Vector<SCALAR,SIZE>& b) {
  SCALAR res(0);
  loop<SIZE>([&res,&a,&b](std::size_t i){ res += a[i]*b[i]; });
  return res;
}
//! Left scalar multiplication.
template< typename SCALAR, std::size_t SIZE >
inline Vector<SCALAR,SIZE> operator*(const SCALAR& a, const Vector<SCALAR,SIZE>& b) {
  Vector<SCALAR,SIZE> c;
  loop<SIZE>([&c,&a,&b](std::size_t i){ c[i] += a*b[i]; });
  return c;
}
//! Right scalar multiplication.
template< typename SCALAR, std::size_t SIZE >
inline Vector<SCALAR,SIZE> operator*(const Vector<SCALAR,SIZE>& a, const SCALAR& b) {
  return b*a;
}
//! Right scalar multiplication.
template< typename SCALAR, std::size_t SIZE >
inline Vector<SCALAR,SIZE> operator/(const Vector<SCALAR,SIZE>& a, const SCALAR& b) {
  return (1./b)*a;
}

// Convenient typedefs.
using Vector3d = Vector<double,3>;
template< std::size_t SIZE >
using VectorXd = Vector<double,SIZE>;

//! Cross-product for 3-dimensional vectors.
inline Vector3d cross(const Vector3d& a, const Vector3d& b) {
  return Vector3d( {a[1]*b[2]-a[2]*b[1],
                    a[2]*b[0]-a[0]*b[2],
                    a[0]*b[1]-a[1]*b[0]} );
}

#endif // VECTOR_HPP
