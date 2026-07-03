#include "compute_born_matrix_nonlinear.h"

#include "atom.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
/* ---------------------------------------------------------------------- */

ComputeBornMatrixNonlinear::ComputeBornMatrixNonlinear(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), values_local(nullptr), values_global(nullptr),
    id_born(nullptr), compute_born(nullptr), temp_x(nullptr), temp_f(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal compute born/matrix/nonlinear command");

  nvalues = NDIR * NBORN; //126

  numdelta = utils::numeric(FLERR, arg[3], false, lmp);
  if (numdelta <= 0.0) error->all(FLERR, "Illegal compute born/matrix command");
  id_born = utils::strdup(arg[4]);
  compute_born = modify->get_compute_by_id(id_born);
  if (!compute_born)
    error->all(FLERR, 4, "Could not find compute born/matrix/nonliear born/matrix ID {}", id_born);

  // Initialize some variables

  values_local = values_global = vector = nullptr;

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
}

/* ---------------------------------------------------------------------- */

ComputeBornMatrixNonlinear::~ComputeBornMatrixNonlinear()
{
  memory->destroy(values_global);
  memory->destroy(vector);
  memory->destroy(temp_x);
  memory->destroy(temp_f);
  delete[] id_born;
}

/* ---------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::init()
{
  // re-check for virial compute

  compute_born = modify->get_compute_by_id(id_born);
  if (!compute_born)
    error->all(FLERR, Error::NOLASTLINE, "Could not find compute born/matrix ID {}",
                id_born);
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

  // store copy of current forces for owned and ghost atoms

  double **x = atom->x;
  double **f = atom->f;

  for (int i = 0; i < nall; i++)
    for (int k = 0; k < 3; k++) {
      temp_x[i][k] = x[i][k];
      temp_f[i][k] = f[i][k];
    }

  // loop over 6 strain directions
  // compute stress finite difference in each direction

  for (int idir = 0; idir < NDIR; idir++) {

    // forward

    displace_atoms(nall, idir, 1.0);
    compute_born->compute_vector();
    for (int j = 0; j < NBORN; j++) {
      values_global[idir * NBORN + j] = compute_born->vector[j];
    }
    restore_atoms(nall, idir);

    // backward

    displace_atoms(nall, idir, -1.0);
    compute_born->compute_vector();
    for (int j = 0; j < NBORN; j++) {
      values_global[idir * NBORN + j] -= compute_born->vector[j];
    }
    restore_atoms(nall, idir);
  }

  // apply derivative factor

  double denominator = 0.5 / numdelta;
  for (int m = 0; m < nvalues; m++) values_global[m] *= denominator;

  // restore original forces for owned and ghost atoms

  for (int i = 0; i < nall; i++)
    for (int k = 0; k < 3; k++) f[i][k] = temp_f[i][k];

  for (int m = 0; m < nvalues; m++) vector[m] = values_global[m];
}


/* ----------------------------------------------------------------------
   displace position of all owned and ghost atoms
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::displace_atoms(int nall, int idir, double magnitude)
{
  double **x = atom->x;

  // NOTE: virial_addon() expressions predicated on
  // shear strain fields (l != k) being symmetric here
  int k = dirlist[idir][0];
  int l = dirlist[idir][1];

  // axial strain

  if (l == k)
    for (int i = 0; i < nall; i++)
      x[i][k] = temp_x[i][k] + numdelta * magnitude * (temp_x[i][l] - fixedpoint[l]);

  // symmetric shear strain

  else
    for (int i = 0; i < nall; i++) {
      x[i][k] = temp_x[i][k] + 0.5 * numdelta * magnitude * (temp_x[i][l] - fixedpoint[l]);
      x[i][l] = temp_x[i][l] + 0.5 * numdelta * magnitude * (temp_x[i][k] - fixedpoint[k]);
      // x[i][l] = temp_x[i][l] + numdelta * magnitude * (temp_x[i][k] - fixedpoint[k]);
    }
}

/* ----------------------------------------------------------------------
   restore position of all owned and ghost atoms
------------------------------------------------------------------------- */

void ComputeBornMatrixNonlinear::restore_atoms(int nall, int idir)
{

  // reset only idir coord

  int k = dirlist[idir][0];
  int l = dirlist[idir][1];
  double **x = atom->x;
  if (l == k)
    for (int i = 0; i < nall; i++) x[i][k] = temp_x[i][k];
  else
    for (int i = 0; i < nall; i++) {
      x[i][l] = temp_x[i][l];
      x[i][k] = temp_x[i][k];
    }
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