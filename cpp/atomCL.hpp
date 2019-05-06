#ifndef ATOMCL_HPP
#define ATOMCL_HPP
// STL includes ----------------------------------------------------------------
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>
// Own includes ----------------------------------------------------------------
#include "vector.hpp"


/*! @brief Provides a structure for the atoms using cell lists.
 *
 *  If periodicity is not set along an axis, the vacuum is assumed outside.
 *  Furthermore, all points must in this case be within the specified cell.
 *  The cutoff radius must be smaller or equal to length of the specified box
 *  otherwise errors may occur.
 *
 *  @note Indices use integers since negative values are expected.
 */
class Atoms {
public:
  //! Type of an index.
  using index_t = int;
  //! 3-tupel of indices.
  using indices_t = std::array<index_t,3>;
  //! Cell list type.
  using list_t = std::vector<index_t>;

private:
  //! All cell lists for atom indices.
  std::vector<list_t> cellLists;
  //! Atom positions.
  const Vector3d *atoms;
  //! Number of atoms.
  const index_t noAtoms;
  //! Flags for periodic boundary conditions.
  const bool *pbc;
  //! Length of box used for cell lists.
  const Vector3d abc;
  //! Number of cells along each axes.
  const indices_t cSize;
  //! Length of a cell.
  const Vector3d cBox;
  //! Squared cutoff radius.
  const double rcut2;

public:

  /*! @brief Constructor.
   *
   *  @note The cutoff should be bigger than the box, otherwise
   *        some atoms may be missed.
   *
   *  @param atoms   C-array of atom positions.
   *  @param noAtoms Number of atoms.
   *  @param rcut    Cutoff radius.
   *  @param pbc     Array containing flags for periodicity.
   *  @param abc     Length of box used (positive, starting from zero).
   */
  Atoms( const Vector3d *atoms, index_t noAtoms, double rcut, 
         const bool *pbc, const Vector3d &abc )
    : atoms(atoms), noAtoms(noAtoms), rcut2(rcut*rcut), pbc(pbc), abc(abc),
      cSize({std::max(static_cast<index_t>(std::floor(abc[0]/rcut)),1),
             std::max(static_cast<index_t>(std::floor(abc[1]/rcut)),1),
             std::max(static_cast<index_t>(std::floor(abc[2]/rcut)),1)}),
      cBox({abc[0]/cSize[0], abc[1]/cSize[1], abc[2]/cSize[2]}) {
    // Fill cell lists
    cellLists.resize(cSize[0]*cSize[1]*cSize[2]);
    for( std::size_t atomIdx = 0; atomIdx < noAtoms; ++atomIdx ) {
      const indices_t ids = wrapIds(getIndices(atoms[atomIdx]));
      cellLists[flatten(ids)].push_back(atomIdx);
    }
  }


  //! @brief Flattens 3-tupel of indices.
  inline std::size_t flatten(const indices_t &id) const {
    return (id[0]*cSize[1]+id[1])*cSize[2]+id[2];
  }
  //! @brief Determines the 3-tupel of indices for a point.
  inline indices_t getIndices( const Vector3d &point ) const {
    return { static_cast<index_t>(std::floor(point[0]/cBox[0])),
             static_cast<index_t>(std::floor(point[1]/cBox[1])),
             static_cast<index_t>(std::floor(point[2]/cBox[2])) };
  }

  //! @brief Wraps an index around an axis.
  inline index_t wrapIdx( index_t ax, index_t id ) const {
    return id < 0 ? wrapIdx(ax, id+cSize[ax]) : id%cSize[ax];
  }
  //! @brief Wraps a 3-tupel of indices.
  inline indices_t wrapIds( indices_t ids ) const {
    return {ids[0] < 0 ? wrapIdx(0, ids[0]+cSize[0]) : ids[0]%cSize[0],
            ids[1] < 0 ? wrapIdx(1, ids[1]+cSize[1]) : ids[1]%cSize[1],
            ids[2] < 0 ? wrapIdx(2, ids[2]+cSize[2]) : ids[2]%cSize[2]};
  }

  /*! @brief Provides the shift of a cell within periodic boundaries
   *
   *  @attention This does not check if box is actually periodic and assumes
   *             it is.
   *
   *  @param ids 3-tupel of indices for cell.
   */
  inline Vector3d shiftVector( indices_t ids ) const {
    Vector3d shift = {0.0,0.0,0.0};
    for( index_t ax = 0; ax < 3; ++ax ) {
      for(; ids[ax] < 0; ids[ax] += cSize[ax])
        shift[ax] -= abc[ax]*pbc[ax];
      for(; ids[ax] >= cSize[ax]; ids[ax] -= cSize[ax])
        shift[ax] += abc[ax]*pbc[ax];
    }
    return shift;
  }


  /*! @brief Applies a function to all atoms in neighbourhood of a point.
   *
   *  @attention The reference point is not restricted to the periodic box.
   *
   *  @param point Reference point.
   *  @param func  Function taking in vector pointing from neighbour to 
   *               reference point and index for atom.
   */
  template< class FUNCTION_T >
  inline void doFor( const Vector3d &point, const FUNCTION_T &func ) const {
    // 3-tupel of indices for point
    const indices_t pointIds = getIndices(point);

    // For all neighbouring cells
    for( index_t x = -1; x <= 1; ++x ) { 
      for( index_t y = -1; y <= 1; ++y ) { 
        for( index_t z = -1; z <= 1; ++z ) { 
          indices_t nIdx = {pointIds[0]+x, pointIds[1]+y, pointIds[2]+z};

          // Check if cell exists
          bool skipIndex = false;
          for( index_t ax = 0; ax < 3; ++ax ) {
            if( !pbc[ax] && (nIdx[ax] >= cSize[ax] || nIdx[ax] < 0) )
              skipIndex = true;
          } // end for ax
          if( skipIndex )
            continue;

          Vector3d shift = shiftVector(nIdx);
          nIdx = wrapIds(nIdx);
          // Evaluate for all atoms in cell
          for( const index_t &atomIdx : cellLists[flatten(nIdx)] ) {
            const Vector3d r = point-atoms[atomIdx]-shift;
            if( r.norm2() < rcut2 ) {
              func(r, atomIdx);
            }
          } // end for atom
        } // end for z
      } // end for y
    } // end for x
  }

}; // end class Atoms

#endif // ATOMCL_HPP
