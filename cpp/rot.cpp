// STL includes ----------------------------------------------------------------
#include <cmath>
#include <cstddef>
// OpenMP includes -------------------------------------------------------------
#include <omp.h>
// Python includes -------------------------------------------------------------
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
// Own includes ----------------------------------------------------------------
#include "vector.hpp"


extern "C" {

  /* @brief Computes COO representation of rotational matrices.
   *
   * @param shiftedGrids List of grids. The grids give the displacement of the
                         apex atoms.
   * @param rowIdx       Array for row-indices (COO).
   * @param colIdx       Array for column-indices (COO).
   * @param dataList     List for rotational matrices (COO).
   */
  void computeRot( PyObject *shiftedGrids,
                   PyArrayObject *rowIdxObject,
                   PyArrayObject *colIdxObject,
                   PyObject *dataList ) {
    // Number of atoms to tunnel to
    const std::size_t noTunnels = PyList_GET_SIZE(shiftedGrids);
    const std::size_t noPoints = PyArray_DIMS((PyArrayObject*) 
      PyList_GET_ITEM(shiftedGrids,0))[0];
    int *rowIdx = (int*) PyArray_DATA(rowIdxObject);
    int *colIdx = (int*) PyArray_DATA(colIdxObject);

    #pragma omp parallel for
    for( std::size_t pointIdx = 0; pointIdx < noPoints; ++pointIdx ) {
      const int curIdx = 9*pointIdx;
      const int pxIdx = pointIdx;
      const int pyIdx = pointIdx+noPoints;
      const int pzIdx = pointIdx+2*noPoints;

      rowIdx[curIdx] =   pxIdx; rowIdx[curIdx+1] = pxIdx; rowIdx[curIdx+2] = pxIdx;
      rowIdx[curIdx+3] = pyIdx; rowIdx[curIdx+4] = pyIdx; rowIdx[curIdx+5] = pyIdx; 
      rowIdx[curIdx+6] = pzIdx; rowIdx[curIdx+7] = pzIdx; rowIdx[curIdx+8] = pzIdx;

      colIdx[curIdx] =   pxIdx; colIdx[curIdx+1] = pyIdx; colIdx[curIdx+2] = pzIdx;
      colIdx[curIdx+3] = pxIdx; colIdx[curIdx+4] = pyIdx; colIdx[curIdx+5] = pzIdx; 
      colIdx[curIdx+6] = pxIdx; colIdx[curIdx+7] = pyIdx; colIdx[curIdx+8] = pzIdx;

      for( std::size_t tunnelIdx = 0; tunnelIdx < noTunnels; ++tunnelIdx ) {
        // Access grid for tunnelIdx atom
        Vector3d *grid1 = (Vector3d*) PyArray_DATA((PyArrayObject*) 
          PyList_GET_ITEM(shiftedGrids,tunnelIdx));
        // Access storage
        double *data = (double *) PyArray_DATA((PyArrayObject*) 
          PyList_GET_ITEM(dataList,tunnelIdx));

        Vector3d v = {0.,0.,-1.};
        Vector3d w = grid1[pointIdx];
        w /= w.norm();
        Vector3d n = cross(v,w);
        if( n.norm() < 1e-12 ) { // Not rotated
          data[curIdx]   = 1.0; data[curIdx+1] = 0.0; data[curIdx+2] = 0.0;
          data[curIdx+3] = 0.0; data[curIdx+4] = 1.0; data[curIdx+5] = 0.0; 
          data[curIdx+6] = 0.0; data[curIdx+7] = 0.0; data[curIdx+8] = 1.0;
        } else {
          n /= n.norm();
          const double cosa = v*w;
          const double sina = std::sqrt(1.0-cosa*cosa);
          data[curIdx]   = n[1]*n[1]*(1-cosa)+cosa; 
          data[curIdx+1] = n[2]*n[1]*(1-cosa)-n[0]*sina;
          data[curIdx+2] = n[0]*n[1]*(1-cosa)+n[2]*sina;
          data[curIdx+3] = n[1]*n[2]*(1-cosa)+n[0]*sina;
          data[curIdx+4] = n[2]*n[2]*(1-cosa)+cosa;
          data[curIdx+5] = n[0]*n[2]*(1-cosa)-n[1]*sina;
          data[curIdx+6] = n[1]*n[0]*(1-cosa)-n[2]*sina;
          data[curIdx+7] = n[2]*n[0]*(1-cosa)+n[1]*sina;
          data[curIdx+8] = n[0]*n[0]*(1-cosa)+cosa;
        }
      } // end for tunnelIdx
    } // end for pointIdx
  } // end computeRot

} // end extern "C"
