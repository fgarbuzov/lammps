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

#include "compute_born_matrix_nonlinear.h"

#include "angle.h"
#include "atom.h"
#include "atom_vec.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

// this table maps the 21 independent Voigt strain pairs
// to the Born vector indices (carbon copy of C_albe from ComputeBornMatrix)

static int constexpr C_albe[21][2] = {
    {0, 0},    // C11
    {1, 1},    // C22
    {2, 2},    // C33
    {3, 3},    // C44
    {4, 4},    // C55
    {5, 5},    // C66
    {0, 1},    // C12
    {0, 2},    // C13
    {0, 3},    // C14
    {0, 4},    // C15
    {0, 5},    // C16
    {1, 2},    // C23
    {1, 3},    // C24
    {1, 4},    // C25
    {1, 5},    // C26
    {2, 3},    // C34
    {2, 4},    // C35
    {2, 5},    // C36
    {3, 4},    // C45
    {3, 5},    // C46
    {4, 5}     // C56
};

/* ---------------------------------------------------------------------- */

ComputeBornMatrixNonlinear::ComputeBornMatrixNonlinear(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), values_global(nullptr),
    id_virial(nullptr), compute_virial(nullptr), temp_x(nullptr), temp_f(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal compute born/matrix/nonlinear command");

  nvalues = NPAIR * NSTRESS; //126
  numflag = 1; // the only option for now
  
  // skip numdiff arg if it exists
  int iarg = 3;
  if (strcmp(arg[iarg], "numdiff") == 0) { iarg++; }

  numdelta = utils::numeric(FLERR, arg[iarg], false, lmp);
  if (numdelta <= 0.0) error->all(FLERR, "Illegal compute born/matrix/nonlinear command");
  
  id_virial = utils::strdup(arg[iarg+1]);
  compute_virial = modify->get_compute_by_id(id_virial);
  if (!compute_virial)
    error->all(FLERR, iarg+1, "Could not find compute born/matrix/nonlinear virial ID {}", 
               id_virial);
  if (compute_virial->pressflag == 0)
    error->all(FLERR, iarg+1, "Compute born/matrix/nonlinear virial ID {} does not compute "
               "pressure", id_virial);

  // Initialize some variables

  values_global = vector = nullptr;

  // this fix produces a global vector

  memory->create(vector, nvalues, "born_matrix_nonlinear:vector");
  memory->create(values_global, nvalues, "born_matrix_nonlinear:values_global");
  size_vector = nvalues;

  vector_flag = 1;
  extvector = 0;
  maxatom = 0;

  reallocate();

  // set fixed-point to default = center of cell

  fixedpoint[0] = 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
  fixedpoint[1] = 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
  fixedpoint[2] = 0.5 * (domain->boxlo[2] + domain->boxhi[2]);

  // define the cartesian indices for each strain (Voigt order)

  dirlist[0][0] = 0;
  dirlist[0][1] = 0;
  dirlist[1][0] = 1;
  dirlist[1][1] = 1;
  dirlist[2][0] = 2;
  dirlist[2][1] = 2;

  dirlist[3][0] = 1;
  dirlist[3][1] = 2;
  dirlist[4][0] = 0;
  dirlist[4][1] = 2;
  dirlist[5][0] = 0;
  dirlist[5][1] = 1;

  // reorder LAMMPS virial vector to Voigt order
  // LAMMPS stress order: xx, yy, zz, xy, xz, yz (for symmetric)
  // Voigt order: xx, yy, zz, yz, xz, xy

  virialVtoV[0] = 0;
  virialVtoV[1] = 1;
  virialVtoV[2] = 2;
  virialVtoV[3] = 5;
  virialVtoV[4] = 4;
  virialVtoV[5] = 3;
}

/* ---------------------------------------------------------------------- */

ComputeBornMatrixNonlinear::~ComputeBornMatrixNonlinear()
{
  memory->destroy(values_global);
  memory->destroy(vector);
  memory->destroy(temp_x);
  memory->destroy(temp_f);
  delete[] id_virial;
}

/* ---------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::init()
{
  // re-check for virial compute

  compute_virial = modify->get_compute_by_id(id_virial);
  if (!compute_virial)
    error->all(FLERR, Error::NOLASTLINE, "Could not find compute born/matrix/nonlinear virial ID {}",
                id_virial);
}


/* ----------------------------------------------------------------------
   compute output vector
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::compute_vector()
{
  invoked_vector = update->ntimestep;

  // grow arrays if necessary

  int nall = atom->nlocal + atom->nghost;
  if (nall > maxatom) reallocate();

  // store copy of current positions and forces for owned and ghost atoms

  double **x = atom->x;
  double **f = atom->f;

  for (int i = 0; i < nall; i++)
    for (int k = 0; k < 3; k++) {
      temp_x[i][k] = x[i][k];
      temp_f[i][k] = f[i][k];
    }

  // loop over 21 independent strain pairs
  // compute second derivative finite difference of virial stress
  // in each strain pair direction

  for (int ipair = 0; ipair < NPAIR; ipair++) {

    int idir1 = C_albe[ipair][0];
    int idir2 = C_albe[ipair][1];

    // (+,+)
    displace_atoms(nall, idir1, idir2, 1.0, 1.0);
    force_clear(nall);
    update_virial();
    for (int jdir = 0; jdir < NSTRESS; jdir++)
      values_global[ipair * NSTRESS + jdir] = compute_virial->vector[virialVtoV[jdir]];
    restore_atoms(nall);

    // (+,-)
    displace_atoms(nall, idir1, idir2, 1.0, -1.0);
    force_clear(nall);
    update_virial();
    for (int jdir = 0; jdir < NSTRESS; jdir++)
      values_global[ipair * NSTRESS + jdir] -= compute_virial->vector[virialVtoV[jdir]];
    restore_atoms(nall);

    // (-,+)
    displace_atoms(nall, idir1, idir2, -1.0, 1.0);
    force_clear(nall);
    update_virial();
    for (int jdir = 0; jdir < NSTRESS; jdir++)
      values_global[ipair * NSTRESS + jdir] -= compute_virial->vector[virialVtoV[jdir]];
    restore_atoms(nall);

    // (-,-)
    displace_atoms(nall, idir1, idir2, -1.0, -1.0);
    force_clear(nall);
    update_virial();
    for (int jdir = 0; jdir < NSTRESS; jdir++)
      values_global[ipair * NSTRESS + jdir] += compute_virial->vector[virialVtoV[jdir]];
    restore_atoms(nall);

    // apply derivative factor
    double denominator = -1.0 / (4.0 * numdelta * numdelta);
    for (int jdir = 0; jdir < NSTRESS; jdir++)
      values_global[ipair * NSTRESS + jdir] *= denominator;
  }

  // add on virial terms (dummy for now)

  virial_addon();

  // convert from pressure to energy units

  double inv_nktv2p = 1.0 / force->nktv2p;
  double volume = domain->xprd * domain->yprd * domain->zprd;
  for (int m = 0; m < nvalues; m++) { values_global[m] *= inv_nktv2p * volume; }

  // restore original forces for owned and ghost atoms

  for (int i = 0; i < nall; i++)
    for (int k = 0; k < 3; k++) f[i][k] = temp_f[i][k];

  // copy to output vector

  for (int m = 0; m < nvalues; m++) vector[m] = values_global[m];
}

/* ----------------------------------------------------------------------
   displace position of all owned and ghost atoms by two simultaneous strains
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::displace_atoms(int nall, int idir1, int idir2,
                                                double magnitude1, double magnitude2)
{
  double **x = atom->x;

  int k = dirlist[idir1][0];
  int l = dirlist[idir1][1];
  int m = dirlist[idir2][0];
  int n = dirlist[idir2][1];

  // NOTE: just as in ComputeBornMatrix, expressions predicated on 
  // shear strain fields (l != k and m != n) being symmetric here
  for (int i = 0; i < nall; i++) {
    x[i][0] = temp_x[i][0];
    x[i][1] = temp_x[i][1];
    x[i][2] = temp_x[i][2];

    // apply strain 1
    if (l == k) {
      x[i][k] += numdelta * magnitude1 * (temp_x[i][l] - fixedpoint[l]);
    } else {
      x[i][k] += 0.5 * numdelta * magnitude1 * (temp_x[i][l] - fixedpoint[l]);
      x[i][l] += 0.5 * numdelta * magnitude1 * (temp_x[i][k] - fixedpoint[k]);
    }

    // apply strain 2 (on top of strain 1!)
    if (n == m) {
      x[i][m] += numdelta * magnitude2 * (x[i][m] - fixedpoint[m]);
    } else {
      x[i][m] += 0.5 * numdelta * magnitude2 * (x[i][n] - fixedpoint[n]);
      x[i][n] += 0.5 * numdelta * magnitude2 * (x[i][m] - fixedpoint[m]);
    }    
  }
}

/* ----------------------------------------------------------------------
   restore position of all owned and ghost atoms
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::restore_atoms(int nall)
{
  double **x = atom->x;
  for (int i = 0; i < nall; i++)
    for (int k = 0; k < 3; k++)
      x[i][k] = temp_x[i][k];
}

/* ----------------------------------------------------------------------
   evaluate potential forces and virial
   carbon copy of the same method in ComputeBornMatrix
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::update_virial()
{
  int eflag = 0;
  int vflag = VIRIAL_PAIR;

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  compute_virial->compute_vector();
}

/* ----------------------------------------------------------------------
   calculate virial stress addon terms to the nonlinear Born matrix
   (dummy for now - to be implemented later)
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::virial_addon()
{
  // placeholder for stress addon terms
}

/* ----------------------------------------------------------------------
   clear forces needed
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::force_clear(int nall)
{
  double **forces = atom->f;
  size_t nbytes = 3 * sizeof(double) * nall;
  if (nbytes) memset(&forces[0][0], 0, nbytes);
}

/* ----------------------------------------------------------------------
   reallocated local per-atoms arrays
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::reallocate()
{
  memory->destroy(temp_x);
  memory->destroy(temp_f);
  maxatom = atom->nmax;
  memory->create(temp_x, maxatom, 3, "born/matrix/nonlinear:temp_x");
  memory->create(temp_f, maxatom, 3, "born/matrix/nonlinear:temp_f");
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double ComputeBornMatrixNonlinear::memory_usage()
{
  double bytes = 0.0;
  bytes += (double) 2 * maxatom * 3 * sizeof(double);
  return bytes;
}