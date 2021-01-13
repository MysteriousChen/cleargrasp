// Include file for system of equation class

////////////////////////////////////////////////////////////////////////
// Just making sure
////////////////////////////////////////////////////////////////////////

#ifndef __RN_SYSTEM_OF_EQUATIONS__
#define __RN_SYSTEM_OF_EQUATIONS__


////////////////////////////////////////////////////////////////////////
// Class definition
////////////////////////////////////////////////////////////////////////
#include <chrono>
#include <fstream>

class RNSystemOfEquations {
public:
  // Constructor/destructor
  RNSystemOfEquations(int nvariables = 0);
  RNSystemOfEquations(const RNSystemOfEquations& system);
  ~RNSystemOfEquations(void);

  // Property functions
  int NVariables(void) const;
  int NEquations(void) const;
  int NPartialDerivatives(void) const;
  RNBoolean IsLinear(void) const;
  RNBoolean IsQuadratic(void) const;
  RNBoolean IsPolynomial(void) const;
  RNBoolean IsAlgebraic(void) const;
  RNBoolean HasVariable(int v) const;

  // Access functions
  RNEquation *Equation(int k) const;
  RNScalar LowerBound(int variable) const;
  RNScalar UpperBound(int variable) const;

  // Manipulation functions
  void InsertEquation(RNPolynomial *polynomial);
  void InsertEquation(RNAlgebraic *algebraic);
  void InsertEquation(RNEquation *equation);
  void RemoveEquation(RNEquation *equation);

  // Variable constraints
  void SetLowerBound(int variable, RNScalar bound);
  void SetUpperBound(int variable, RNScalar bound);
  void RemoveLowerBound(int variable);
  void RemoveUpperBound(int variable);

  // Evaluation functions
  void EvaluateResiduals(const RNScalar *x, RNScalar *y) const;
  RNScalar SumOfSquaredResiduals(const RNScalar *x) const;

  // Optimization functions
  int Minimize(RNScalar *x, int solver = 0, RNScalar tolerance = RN_EPSILON) const;

  // Print functions
  void PrintEquations(FILE *fp = stdout) const;
  void PrintValues(const RNScalar *x, FILE *fp = stdout) const;
  void PrintResiduals(const RNScalar *x, FILE *fp = stdout) const;
  void PrintPartialDerivatives(const RNScalar *x, FILE *fp = stdout) const;
  void Print(FILE *fp = stdout) const;

public:
  // Do not use these
  void InsertEquation(RNPolynomial *polynomial, RNScalar residual_threshold);
  void InsertEquation(RNAlgebraic *algebraic, RNScalar residual_threshold);
  
public:
  int *index_to_variable;
  int *variable_to_index;
  int *variable_marks;
  int current_mark;

public:
  RNScalar *lower_bounds;
  RNScalar *upper_bounds;

private:
  int nvariables;
  RNArray<RNEquation *> equations;
};



////////////////////////////////////////////////////////////////////////
// Inline functions 
////////////////////////////////////////////////////////////////////////

inline int RNSystemOfEquations::
NVariables(void) const
{
  // Return number of variables
  return nvariables;
}



inline int RNSystemOfEquations::
NEquations(void) const
{
  // Return number of equations
  return equations.NEntries();
}



inline RNEquation *RNSystemOfEquations::
Equation(int k) const
{
  // Return Kth equation
  return equations.Kth(k);
}



inline RNScalar RNSystemOfEquations::
LowerBound(int variable) const
{
  // Return lower bound on variable (or FLT_MAX if there is none)
  assert((variable >= 0) && (variable < nvariables));
  if (!lower_bounds) return FLT_MAX;
  return lower_bounds[variable];
}



inline RNScalar RNSystemOfEquations::
UpperBound(int variable) const
{
  // Return upper bound on variable (or FLT_MAX if there is none)
  assert((variable >= 0) && (variable < nvariables));
  if (!upper_bounds) return FLT_MAX;
  return upper_bounds[variable];
}



inline void RNSystemOfEquations::
RemoveLowerBound(int variable)
{
  // Remove bound
  SetLowerBound(variable, -FLT_MAX);
}



inline void RNSystemOfEquations::
RemoveUpperBound(int variable)
{
  // Remove upper bound
  SetUpperBound(variable, FLT_MAX);
}



////////////////////////////////////////////////////////////////////////
// System of equation solvers
////////////////////////////////////////////////////////////////////////

enum {
  RN_AMGCL_SOLVER,
  RN_CERES_SOLVER,
  RN_MINPACK_SOLVER,
  RN_SPLM_SOLVER,
  RN_CSPARSE_SOLVER,
  RN_NUM_SOLVERS
};



////////////////////////////////////////////////////////////////////////
// Select dependencies (can be set in compile flags by app)
////////////////////////////////////////////////////////////////////////

// #define RN_USE_SPLM
// #define RN_NO_SPLM
// #define RN_USE_MINPACK
// #define RN_NO_MINPACK
// #define RN_USE_CERES
// #define RN_NO_CERES
// #define RN_USE_CSPARSE
// #define RN_NO_CSPARSE
#define RN_USE_AMGCL

////////////////////////////////////////////////////////////////////////
// AMGCL Stuff
////////////////////////////////////////////////////////////////////////
#ifdef RN_NO_AMGCL
#undef RN_USE_AMGCL
#endif
#ifdef RN_USE_AMGCL
// New using amgcl library
#include "CSparse/CSparse.h"
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <iostream>
#include <vector>
#include <cmath>

typedef amgcl::backend::builtin<double> Backend;
//typedef amgcl::backend::vexcl<double> Backend;

typedef amgcl::make_solver<
    amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
        >,
    amgcl::solver::bicgstabl<Backend>
    > Solver;

static
void Csparse2AMGCL(cs* a, std::vector<int> &ptr, std::vector<int> &col, std::vector<double> &val){
  
  // CSC to CSR 
  int m = a->m;
  int nz = a->nzmax;
  int *p = a->p;
  int *i = a->i;
  double *x = a->x;

  // initial memory space
  ptr.resize(m+1);
  col.resize(nz);
  val.resize(nz);

  for(int idx=0; idx<nz; ++idx){
    ptr[i[idx]]++;
  }

  //cumsum the nnz per column to get ptr
  int cumsum = 0;
  for(int _col=0; _col<m; ++_col){
    int temp = ptr[_col];
    ptr[_col] = cumsum;
    cumsum += temp;
  }
  ptr[m] = nz;

  for(int row=0; row<m; ++row){
    for(int jj=p[row]; jj<p[row+1]; jj++){
      int _col = i[jj];
      int dest = ptr[_col];

      col[dest] = row;
      val[dest] = x[jj];

      ptr[_col]++;
    }
  }

  int last = 0;
  for(int _col=0; _col<=m; ++_col){
    int temp = ptr[_col];
    ptr[_col] = last;
    last = temp;
  }

}


static int 
MinimizeAMGCL(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{

  //std::ofstream ansfile, myfile;
  //ansfile.open("Result.txt");
  //myfile.open("SparseMat.txt");
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  begin = std::chrono::steady_clock::now();
  // Get convenient variables
  const int n = system->NVariables();
  const int mm = system->NEquations();
  const int max_nz = system->NPartialDerivatives();

  end = std::chrono::steady_clock::now();
  std::cout << "Initial Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  // Formulate problem into compress sparse matrix  
  std::vector<int>    ptr, col;
  std::vector<double> val, rhs;

  ptr.clear();
  col.clear();
  val.clear();
  rhs.resize(mm);

  begin = std::chrono::steady_clock::now();
  // Allocate matrix
  cs *a = cs_spalloc (0, n, max_nz, 1, 1);
  if (!a) {
    fprintf(stderr, "Unable to allocate cs matrix: %d %d\n", n, max_nz);
    return 0;
  }

  double *b = new double [ mm ];
  for (int i = 0; i < mm; i++) b[i] = 0;
  
  // Temporary information 
  double *x_temp = new double [ n ];
  for (int i = 0; i < n; i++) x_temp[i] = 0;

  double *lhs = new double [ n ];
  for (int i = 0; i < n; i++) lhs[i] = 0;
  end = std::chrono::steady_clock::now();
  std::cout << "Setup Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  begin = std::chrono::steady_clock::now();
  // Fill matrix
  int m = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    // Initialize constant term
    double rhs_val = -equation->Evaluate(x_temp);
    // Mark variables in equation
    int nz = 0;
    int variable_count = 0;
    RNSystemOfEquations *tmp = (RNSystemOfEquations *) system;
    equation->UpdateVariableIndex(n, variable_count, tmp->variable_marks, tmp->current_mark++, tmp->index_to_variable);
    for (int j = 0; j < variable_count; j++) {
      int v = tmp->index_to_variable[j];
      lhs[v] = equation->PartialDerivative(x_temp, v);
      if (lhs[v] != 0) nz++;
    }

    // Add data to matrix if there are nonzero entries 
    if (nz > 0) {
      assert(m < mm);
      for (int j = 0; j < variable_count; j++) {
        int v = tmp->index_to_variable[j];
        if (lhs[v] == 0) continue;
        cs_entry(a, m, v, lhs[v]);
      }
      b[m] = rhs_val;
      m++;
    }
  }
  // Just checking
  assert(a->m == m);
  assert(a->n == n);
  assert(a->n == system->NVariables());
  assert(a->m <= system->NEquations());
  assert(a->nz <= max_nz);
  end = std::chrono::steady_clock::now();
  std::cout << "Fill Matrix = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  begin = std::chrono::steady_clock::now();
  // Setup aT * a * x = aT * b        
  cs *A = cs_compress(a);
  assert(A);
  cs *AT = cs_transpose (A, 1);
  assert(AT);
  cs *ATA = cs_multiply (AT, A);
  assert(ATA);
  cs_gaxpy(AT, b, x_temp);

  end = std::chrono::steady_clock::now();
  std::cout << "Compute Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Convert Csparse to Amgcl
  begin = std::chrono::steady_clock::now();
  Csparse2AMGCL(ATA, ptr, col, val);
  assert(rhs.size() == m);
  assert(ptr.size() == mm+1);
  assert(col.size() == max_nz);
  assert(val.size() == max_nz);

  for(int i=0; i<mm; ++i){
    rhs[i] = x_temp[i];
  }
  end = std::chrono::steady_clock::now();
  std::cout << "Convert Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  begin = std::chrono::steady_clock::now();
  // Delete stuff
  cs_spfree(A);
  cs_spfree(AT);
  cs_spfree(ATA);
  cs_spfree(a);
  delete [] b;
  delete [] x_temp;
  delete [] lhs;
  end = std::chrono::steady_clock::now();
  std::cout << "Free = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Solve by amgcl library
  std::vector<double> x(n, 0.0);
  Solver::params prm;
  prm.solver.tol = tolerance;
  prm.solver.maxiter = 10;

  begin = std::chrono::steady_clock::now();
  Solver solve(std::tie(n, ptr, col, val), prm);
  end = std::chrono::steady_clock::now();
  std::cout << "Prepare = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  int iters = 0;
  double error = 0.0;
  begin = std::chrono::steady_clock::now();
  std::tie(iters, error) = solve(rhs, x);
  for (int i = 0; i < n; i++) io[i] = x[i];
  end = std::chrono::steady_clock::now();
  std::cout << "Solver time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  std::cout<<iters<<","<<error<<"\n";
  std::cout<<"------------------------------------\n";
  return 1;
}


#else

static int 
MinimizeAMGCL(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: Amgcl solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_AMGCL and -lamgcl xxx to compilation and link commands.\n");
  return 0;
}

#endif

////////////////////////////////////////////////////////////////////////
// CERES Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_CERES
#undef RN_USE_CERES
#endif
#ifdef RN_USE_CERES

#include "ceres/ceres.h"

class CeresCostFunction : public ceres::CostFunction {
private:
  RNEquation *equation;
  int nvariables;
public:
  CeresCostFunction(RNEquation *equation = NULL, int nvariables = 0) 
    : equation(equation), nvariables(nvariables)
  {
    set_num_residuals(1);
    for (int i = 0; i < nvariables; i++) {
      mutable_parameter_block_sizes()->push_back(1);
    }
  };

  virtual ~CeresCostFunction(void) 
  {
  };

  virtual bool Evaluate(double const* const* x, double* residual, double** jacobian) const 
  {
    // Evaluate residual
    if (residual != NULL) {
      residual[0] = equation->Evaluate(x);
    }

    // Evaluate Jacobian, if asked for.
    if (jacobian != NULL) {
      for (int v = 0; v < nvariables; v++) {
        jacobian[v][0] = equation->PartialDerivative(x, v);
      }

#if 0
      // Check versus numerical partial derivative
      for (int v = 0; v < nvariables; v++) {
        double tmp0, tmp1, tmp2;
        double **xp = (double **) x;
        tmp0 = x[v][0];
        xp[v][0] = tmp0 + 0.001;
        Evaluate(x, &tmp1, NULL);
        xp[v][0] = tmp0 - 0.001;
        Evaluate(x, &tmp2, NULL);
        xp[v][0] = tmp0;
        double dydx = (tmp1 - tmp2) / 0.002;
        if (RNIsNotEqual(dydx, jacobian[v][0])) {
          printf("PD %d : %g %g : %g %g\n", v, tmp1, tmp2, dydx, jacobian[v][0]);
        }
      }
#endif
    }

    // Return success
    return true;
  }
};



static int 
MinimizeCERES(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  int n = system->NVariables();

  // Allocate ceres stuff
  ceres::Problem *problem = new ceres::Problem();
  ceres::Solver::Options *options = new ceres::Solver::Options();
  ceres::Solver::Summary *summary = new ceres::Solver::Summary();

  // Copy system of equations (so can remap variables) !!!
  RNSystemOfEquations system_copy(*system);

  // Allocate and initialize parameter data
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = io[i];

  // Create ceres residual blocks
  for (int i = 0; i < system_copy.NEquations(); i++) {
    RNEquation *equation = system_copy.Equation(i);

    // Remap variables
    int variable_count = 0;
    equation->UpdateVariableIndex(system_copy.NVariables(), variable_count, 
      system_copy.variable_marks, system_copy.current_mark++, 
      system_copy.index_to_variable, system_copy.variable_to_index, TRUE);
    if (variable_count == 0) continue;

    // Create cost function and residual block stuff
    std::vector<double *> variable_ptr;
    for (int j = 0; j < variable_count; j++) {
      int v = system_copy.index_to_variable[j];
      variable_ptr.push_back(&x[v]);
    }

    // Create cost function
    ceres::CostFunction *cost_function = new CeresCostFunction(equation, variable_count);

    // Create loss function
    ceres::LossFunction *loss_function = NULL;
    if (equation->ResidualThreshold() > 0) {
      loss_function = new ceres::HuberLoss(equation->ResidualThreshold());
    }

    // Add residual block
    problem->AddResidualBlock(cost_function, loss_function, variable_ptr);
  }

  // Set lower bounds
  if (system->lower_bounds) {
    for (int i = 0; i < n; i++) {
      if (system->lower_bounds[i] == -FLT_MAX) continue;
      problem->SetParameterLowerBound(&x[i], 0, system->lower_bounds[i]);
    }
  }
  
  // Set upper bounds
  if (system->upper_bounds) {
    for (int i = 0; i < n; i++) {
      if (system->upper_bounds[i] == -FLT_MAX) continue;
      problem->SetParameterUpperBound(&x[i], 0, system->upper_bounds[i]);
    }
  }
  
  // Run the solver
  // options->max_num_iterations = 128;
  options->num_threads = 12;
  options->num_linear_solver_threads = 12; 
  // options->check_gradients = true;
  // options->gradient_check_relative_precision = 1E-1;
  // options->numeric_derivative_relative_step_size = 1E-3;
  // options->trust_region_strategy_type = ceres::DOGLEG;
  options->linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  
  options->max_solver_time_in_seconds = 12 * 60 * 60; 
  options->function_tolerance = tolerance;
  // options->gradient_tolerance = 1E-12;
  // options->parameter_tolerance = 1E-12;
  // options->min_relative_decrease = 1E-6;
  // options->use_nonmonotonic_steps = true;
  options->minimizer_progress_to_stdout = true;
  Solve(*options, problem, summary);

  // Print report
  // std::cout << summary->FullReport() << "\n";

  // Copy solved solution into result 
  for (int i = 0; i < n; i++) io[i] = x[i]; 

  // Delete data
  delete [] x;

  // Delete ceres stuff
  delete summary;
  delete options;
  delete problem;

  // Return success
  return 1;
}

#else

static int 
MinimizeCERES(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: Ceres solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_CERES and -lceres xxx to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// SPLM Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_SPLM
#undef RN_USE_SPLM
#endif
#ifdef RN_USE_SPLM

#include "splm/splm.h"

static void
SPLMFunction(double *x, double *y, int n, int m, void *data)
{
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // n is number of variables
  // m is number of equations
  // data is a user-data variable

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());

  // Evaluate residuals
  system->EvaluateResiduals(x, y);
}



static void
SPLMJacobian(double *x, struct splm_ccsm *jac, int n, int m, void *data)
{
  // p is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // jac needs to be filled in with non-zero elements of jacobian

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());
  assert(jac->nnz == system->NPartialDerivatives());

  // Allocate triplets
  splm_stm sm;
  splm_stm_allocval(&sm, m, n, jac->nnz);

#if 0
  // Fill triplets
  int ntriplets = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    for (int v = 0; v < n; v++) {
      if (equation->HasVariable(v)) {
        RNScalar d = equation->PartialDerivative(x, v);
        splm_stm_nonzeroval(&sm, i, v, d);
        ntriplets++;
      }
    }
  }
#else
  int ntriplets = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    int count = 0;
    equation->UpdateVariableIndex(system->NVariables(), count, 
      system->variable_marks, system->current_mark++, 
      system->index_to_variable);
    for (int j = 0; j < count; j++) {
      int v = system->index_to_variable[j];
      RNScalar d = equation->PartialDerivative(x, v);
      splm_stm_nonzeroval(&sm, i, v, d);
      ntriplets++;
    }
  }      
#endif

  // Just checking
  if (ntriplets != jac->nnz) {
    fprintf(stderr, "Mismatching number of derivatives: %d %d\n", ntriplets, jac->nnz);
    abort();
  }

  // Convert from triplets to CSM
  splm_stm2ccsm(&sm, jac);

  // Free triplets
  splm_stm_free(&sm);
}



static int 
MinimizeSPLM(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  const int n = system->NVariables();
  const int m = system->NEquations();
  const int jnnz = system->NPartialDerivatives();

  // Allocate temporary data
  double *x = new double [ n ];
  double *y = new double [ m ];

  // Initialize values and residuals
  for (int i = 0; i < n; i++) x[i] = io[i];
  for (int i = 0; i < m; i++) y[i] = 0;

  // Set options to control tolerance
  // double opts[SPLM_OPTSSZ];
  // ???
  
  // Run the solver
  double info[SPLM_INFO_SZ];
  int status = sparselm_derccs(SPLMFunction, SPLMJacobian, x, y, n, 0, m, jnnz, -1, 16, NULL, info, (void *) system);
  if (status == SPLM_ERROR) fprintf(stderr, "Error in SPLM solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

#if 0
  // Print debug info
  printf("SPLM Info: ");
  for (int i = 0; i < SPLM_INFO_SZ; i++) 
    printf("%g ", info[i]);
  printf("\n");
#endif

  // Delete temporary data
  delete [] x;
  delete [] y;

  // Return status 
  return (status == SPLM_ERROR) ? 0 : 1;
}

#else

static int 
MinimizeSPLM(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: SPLM solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_SPLM and -lsplm to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// MINPACK STUFF
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_MINPACK
#undef RN_USE_MINPACK
#endif
#ifdef RN_USE_MINPACK

#include "minpack/minpack.h"


static int 
MinpackFunction(void *data, int m, int n, const double *x, double *y, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // return a negative value to terminate

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());

  // Evaluate function
  system->EvaluateResiduals(x, y);

  // Return success
  return 1;
}



#define USE_LMDER1
#ifdef USE_LMDER1

static int 
MinpackJacobian(void *data, int m, int n, const double *x, double *jacobian, int ldjacobian, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // return a negative value to terminate

  // Get convenient variables
  RNSystemOfEquations *system = (RNSystemOfEquations *) data;
  assert(m == system->NEquations());
  assert(n == system->NVariables());
  assert(ldjacobian == m);

#if 0
  // Evaluate jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    for (int v = 0; v < n; v++) {
      RNScalar d = equation->PartialDerivative(x, v);
      jacobian[v*ldjacobian+i] += term->PartialDerivative(x, v);
    }
  }
#else
  // Initialize jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    for (int v = 0; v < n; v++) {
      jacobian[v*ldjacobian+i] = 0;
    }
  }
  // Evaluate jacobian
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);
    int count = 0;
    equation->UpdateVariableIndex(system->NVariables(), count, 
      system->variable_marks, system->current_mark++, 
      system->index_to_variable);
    for (int j = 0; j < count; j++) {
      int v = system->index_to_variable[j];
      RNScalar d = equation->PartialDerivative(x, v);
      jacobian[v*ldjacobian+i] = d;
    }
  }  
#endif    

  // Return success
  return 1;
}



static int 
MinpackCallback(void *data, int m, int n, const double *x, double *y, double *jacobian, int ldjacobian, int iflag)
{
  // data is a user-data variable
  // m is number of equations
  // n is number of variables
  // x is vector of length n with current variable values
  // y is vector of length m with returned function values
  // jacobian is vector of length m*n with returned jacobian values
  // if iflag=1 fill in y, else if iflag=2 fill in jacobian
  // return a negative value to terminate

  // Call appropriate function
  if (iflag == 2) return MinpackJacobian(data, m, n, x, jacobian, ldjacobian, iflag);
  else return MinpackFunction(data, m, n, x, y, iflag);

  // Return success
  return 1;
}

#endif



static int 
MinimizeMINPACK(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Get convenient variables
  const int m = system->NEquations();
  const int n = system->NVariables();

  // Allocate temporary data
  double *x = new double [ n ];
  double *y = new double [ m ];

  // Initialize values and residuals
  for (int i = 0; i < n; i++) x[i] = io[i];
  for (int i = 0; i < m; i++) y[i] = 0;

#ifdef USE_LMDER1
  // Allocate temporary data
  int lwa = 5*n+m;
  double *wa = new double [ lwa ];
  double *jacobian = new double [ m * n ];
  int *ipvt = new int [ n ];
  double tol = tolerance;   // 1E-3; // sqrt(dpmpar(1));

  // Run the solver
  int status = lmder1(MinpackCallback, (void *) system, m, n, x, y, jacobian, m, tol, ipvt, wa, lwa);
  if (status == 0) fprintf(stderr, "Error in Minpack solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete temporary data
  delete [] ipvt;
  delete [] jacobian;
  delete [] wa;
#else
  // Allocate temporary data
  int lwa = m*n+5*n+m;
  int *iwa = new int [ n ];
  double *wa = new double [ lwa ];
  double tol = 1E-3; // sqrt(dpmpar(1));

  // Run the solver
  int status = lmdif1(MinpackFunction, (void *) system, m, n, x, y, tol, iwa, wa, lwa);
  if (status == 0) fprintf(stderr, "Error in Minpack solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete temporary data
  delete [] iwa;
  delete [] wa;
#endif

  // Delete values and residuals
  delete [] x;
  delete [] y;

  // Return status 
  return status;
}

#else

static int 
MinimizeMINPACK(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: Minpack solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_MINPACK and -lminpack to compilation and link commands.\n");
  return 0;
}

#endif



////////////////////////////////////////////////////////////////////////
// CSPARSE Stuff
////////////////////////////////////////////////////////////////////////

#ifdef RN_NO_CSPARSE
#undef RN_USE_CSPARSE
#endif
#ifdef RN_USE_CSPARSE

#include "CSparse/CSparse.h"

//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
//std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
static int 
MinimizeCSPARSE(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // Get convenient variables
  const int n = system->NVariables();
  const int mm = system->NEquations();
  const int max_nz = system->NPartialDerivatives();

  // Allocate matrix
  cs *a = cs_spalloc (0, n, max_nz, 1, 1);
  if (!a) {
    fprintf(stderr, "Unable to allocate cs matrix: %d %d\n", n, max_nz);
    return 0;
  }
  // Allocate B vector
  double *b = new double [ mm ];
  for (int i = 0; i < mm; i++) b[i] = 0;

  // Allocate X vector
  double *x = new double [ n ];
  for (int i = 0; i < n; i++) x[i] = 0;

  // Allocate temporary data for rows
  double *lhs = new double [ n ];
  for (int i = 0; i < n; i++) lhs[i] = 0;
  end = std::chrono::steady_clock::now();
  std::cout << "Initial time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  begin = std::chrono::steady_clock::now();
  // Fill matrix
  int m = 0;
  for (int i = 0; i < system->NEquations(); i++) {
    RNEquation *equation = system->Equation(i);

    // Initialize constant term
    double rhs = -equation->Evaluate(x);
    // Mark variables in equation
    int nz = 0;
    int variable_count = 0;
    RNSystemOfEquations *tmp = (RNSystemOfEquations *) system;
    equation->UpdateVariableIndex(n, variable_count, tmp->variable_marks, tmp->current_mark++, tmp->index_to_variable);
    for (int j = 0; j < variable_count; j++) {
      int v = tmp->index_to_variable[j];
      lhs[v] = equation->PartialDerivative(x, v);
      if (lhs[v] != 0) nz++;
    }

    // Add data to matrix if there are nonzero entries 
    if (nz > 0) {
      assert(m < mm);
      for (int j = 0; j < variable_count; j++) {
        int v = tmp->index_to_variable[j];
        if (lhs[v] == 0) continue;
        cs_entry(a, m, v, lhs[v]);
      }
      b[m] = rhs;
      m++;
    }
  }
  end = std::chrono::steady_clock::now();
  std::cout << "Fill Matrix time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  begin = std::chrono::steady_clock::now();
  // Just checking
  assert(a->m == m);
  assert(a->n == n);
  assert(a->n == system->NVariables());
  assert(a->m <= system->NEquations());
  assert(a->nz <= max_nz);// system->NPartialDerivatives());

  // Setup aT * a * x = aT * b        
  cs *A = cs_compress(a);
  assert(A);
  cs *AT = cs_transpose (A, 1);
  assert(AT);
  cs *ATA = cs_multiply (AT, A);
  assert(ATA);
  cs_gaxpy(AT, b, x);
  //std::cout<<x[0];
  end = std::chrono::steady_clock::now();
  std::cout << "Compute time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  //cs_print(ATA, 0);
  // Solve linear system
  // int status = cs_lusol (1, ATA, x, RN_EPSILON);
  begin = std::chrono::steady_clock::now();
  int status = cs_cholsol (1, ATA, x);
  end = std::chrono::steady_clock::now();
  std::cout << "Solver time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
  if (status == 0) fprintf(stderr, "Error in CSPARSE solver\n");
  else { for (int i = 0; i < n; i++) io[i] = x[i]; }

  // Delete stuff
  cs_spfree(A);
  cs_spfree(AT);
  cs_spfree(ATA);
  cs_spfree(a);
  delete [] b;
  delete [] x;
  delete [] lhs;

  std::cout<<"------------------------------------\n";

  // Return status
  return status;
}

#else

static int 
MinimizeCSPARSE(const RNSystemOfEquations *system, RNScalar *io, RNScalar tolerance)
{
  // Print error message
  fprintf(stderr, "Cannot minimize equation: CSparse solver disabled during compile.\n");
  fprintf(stderr, "Enable it by adding -DRN_USE_CSPARSE and -lCSparse to compilation and link commands.\n");
  return 0;
}

#endif




inline int RNSystemOfEquations::
Minimize(RNScalar *x, int solver, RNScalar tolerance) const
{
  // Check solver
  if (solver == RN_SPLM_SOLVER) return MinimizeSPLM(this, x, tolerance);
  else if (solver == RN_MINPACK_SOLVER) return MinimizeMINPACK(this, x, tolerance);
  else if (solver == RN_CERES_SOLVER) return MinimizeCERES(this, x, tolerance);
  else if (solver == RN_CSPARSE_SOLVER) return MinimizeCSPARSE(this, x, tolerance);
  else if (solver == RN_AMGCL_SOLVER) return MinimizeAMGCL(this, x, tolerance);
  fprintf(stderr, "System of equation solver not recognized: %d\n", solver);
  return 0;
}



#endif
