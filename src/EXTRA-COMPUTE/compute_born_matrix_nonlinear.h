#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(born/matrix/nonlinear,ComputeBornMatrixNonlinear);
// clang-format on
#else

#ifndef LMP_COMPUTE_BORN_MATRIX_NONLIN_H
#define LMP_COMPUTE_BORN_MATRIX_NONLIN_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeBornMatrixNonlinear : public Compute {
 public:
  ComputeBornMatrixNonlinear(class LAMMPS *, int, char **);
  ~ComputeBornMatrixNonlinear() override;
  void init() override;
  void compute_vector() override;
  double memory_usage() override;

 private:
  // Born matrix contributions

  void displace_atoms(int, int, double);    // displace atoms
  void restore_atoms(int, int);             // restore atom positions
  void reallocate();                        // grow the atom arrays

  int nvalues;        // length of elastic tensor
  double numdelta;    // size of finite strain
  int maxatom;        // allocated size of atom arrays

  double *values_local, *values_global;

  char *id_born;                  // name of virial compute
  class Compute *compute_born;    // pointer to virial compute

  static constexpr int NBORN = 21;
  static constexpr int NDIR  = 6;    // dimension of virial and strain vectors
  static constexpr int NXYZ  = 3;    // number of Cartesian coordinates
  double **temp_x;                   // original coords
  double **temp_f;                   // original forces
  double fixedpoint[NXYZ];    // displacement field origin
  int dirlist[NDIR][2];       // strain cartesian indices
};
}    // namespace LAMMPS_NS

#endif
#endif
