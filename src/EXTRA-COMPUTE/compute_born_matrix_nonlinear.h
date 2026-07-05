/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

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
  // Born matrix nonlinear contributions

  void displace_atoms(int, int, int, double, double);  // displace atoms with two strains
  void force_clear(int);                               // zero out force array
  void update_virial();                                // recalculate the virial
  void restore_atoms(int);                             // restore atom positions
  void born_addon();                                   // Born matrix addon terms
  void virial_addon();
  void reallocate();                                   // grow the atom arrays

  int nvalues;        // length of output vector (126)
  int numflag;        // 1 if using finite differences (the only option for now)
  double numdelta;    // size of finite strain
  int maxatom;        // allocated size of atom arrays

  double *values_global;

  char *id_virial;                  // name of virial compute
  class Compute *compute_virial;    // pointer to virial compute

  char *id_born;                        // name of Born matrix compute
  class Compute *compute_born; // pointer to Born matrix compute

  static constexpr int NSTRESS = 6;     // number of stress components (Voigt)
  static constexpr int NPAIR   = 21;    // number of independent strain pairs
  static constexpr int NDIR    = 6;     // dimension of virial and strain vectors
  static constexpr int NXYZ    = 3;     // number of Cartesian coordinates

  double **temp_x;                   // original coords
  double **temp_f;                   // original forces
  double fixedpoint[NXYZ];           // displacement field origin
  int dirlist[NDIR][2];              // strain cartesian indices
  int virialVtoV[NDIR];              // LAMMPS virial -> Voigt order mapping
  int revalbe_sigma[NXYZ][NXYZ];
  int revalbe_C[NDIR][NDIR];
  int revalbemunu[NXYZ][NXYZ][NXYZ][NXYZ];
};

}    // namespace LAMMPS_NS

#endif
#endif