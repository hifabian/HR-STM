// Purpose: Test correctness of atomCL class.
// Result: Appears to work correctly
#include <iostream>
#include <vector>

#include "../cpp/vector.hpp"
#include "../cpp/atomCL.hpp"



int main() {

  double data[15] = {1.20,0,0.5, -0.9,0,0.5,  0,-0.1,0.5,  0.5,0.5,0.5, 0.0,1.20,0.5};
  bool pbc[3] = {1,1,0};
  const Vector3d abc = {1,1,1};
  Atoms atoms((const Vector3d*) data, 5, 0.45, pbc, abc);

  const auto func = [] (Vector3d r, int a) {
    std::cout << "Atom " << a << " with a distance of (" 
              << r[0] << ", " << r[1] << ", " << r[2] << ")\n";
  };

  const Vector3d p = {0.75, 0, 0.5};
  atoms.doFor(p,func);
  std::cout << "<<<<<<<<<<\n";
  const Vector3d q = {0, 0.75, 0.5};
  atoms.doFor(q,func);

  return 0;
}
