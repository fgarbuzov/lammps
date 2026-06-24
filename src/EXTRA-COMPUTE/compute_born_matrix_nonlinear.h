/* -*- c++ -*- ----------------------------------------------------------
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

#ifndef LMP_COMPUTE_BORN_MATRIX_NONLINEAR_H
#define LMP_COMPUTE_BORN_MATRIX_NONLINEAR_H

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
  void displace_atoms(int, int, double);
  void restore_atoms(int, int);
  void reallocate();

  int nvalues;           // length of nonlinear Born tensor (126 = 6 * 21)
  double numdelta;       // strain magnitude for outer finite difference
  int maxatom;           // allocated size of atom arrays

  double *values_global;   // 126-component result (6 strain dirs x 21 elastic constants)

  char *id_born;                  // name of born matrix compute
  class Compute *compute_born;    // pointer to born matrix compute

  static constexpr int NDIR_VIRIAL = 6;     // number of Voigt strain directions
  static constexpr int NELASTIC = 21;       // number of independent elastic constants
  static constexpr int NXYZ_VIRIAL = 3;     // number of Cartesian coordinates

  int dirlist[NDIR_VIRIAL][2];              // cartesian indices for each strain direction
  double fixedpoint[NXYZ_VIRIAL];           // displacement field origin
  double **temp_x;                          // original coords
  double **temp_f;                          // original forces
};

}    // namespace LAMMPS_NS

#endif
#endif