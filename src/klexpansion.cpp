
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/base/utilities.h>

#include <deal.II/lac/slepc_solver.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <random>
#include <mpi.h>

using namespace dealii;
class ParallelKL
{
public:
  ParallelKL();
  void run();
private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void create_random_field();
  void stats_for_random_field();
  void output_results() const;
  void output_results_parallel();

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<2> triangulation;

  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;

  IndexSet         locally_owned_dofs;
  IndexSet         locally_relevant_dofs;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;

  PETScWrappers::MPI::SparseMatrix system_mass_matrix;
  PETScWrappers::MPI::SparseMatrix system_stiffness_matrix;

  PETScWrappers::MPI::Vector randomfield_vector;

  std::vector<double> eigenvalues;
  std::vector<PETScWrappers::MPI::Vector> eigenvectors;

  std::vector<double> normalized_gaussian;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution;

  unsigned int this_mpi_process, n_mpi_processes;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};

ParallelKL::ParallelKL()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<2>::MeshSmoothing(
                  Triangulation<2>::smoothing_on_refinement |
                  Triangulation<2>::smoothing_on_coarsening))
  , fe(1)
  , dof_handler(triangulation)
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , pcout(std::cout,
        (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
{}

void ParallelKL::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(5);

  //parallel console output
  pcout << "Number of active cells: " << triangulation.n_active_cells()
        << std::endl;
}

void ParallelKL::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_stiffness_matrix.reinit(locally_owned_dofs,
                                 locally_owned_dofs,
                                 dsp,
                                 mpi_communicator);

  system_mass_matrix.reinit(locally_owned_dofs,
                            locally_owned_dofs,
                            dsp,
                            mpi_communicator);

  randomfield_vector.reinit(locally_owned_dofs, mpi_communicator);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  constraints.close();
}

void ParallelKL::assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  system_stiffness_matrix = 0.;
  system_mass_matrix = 0.;

  QGauss<2> quadrature_formula(fe.degree + 1);

  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    { if (cell->is_locally_owned())
       {
          fe_values.reinit(cell);
          cell_mass_matrix = 0.;
          cell_stiffness_matrix = 0.;

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              Point<2> quad_pointq = fe_values.quadrature_point(q_index);
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_mass_matrix(i, j) +=
                    (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                    fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                    fe_values.JxW(q_index));           // dx

              for (const unsigned int l_index : fe_values.quadrature_point_indices())
                {
                  Point<2> quad_pointl = fe_values.quadrature_point(l_index);
                  const double point_distance = quad_pointl.distance(quad_pointq);

                  for (const unsigned int i : fe_values.dof_indices())
                    for (const unsigned int j : fe_values.dof_indices())
                        cell_stiffness_matrix(i, j) +=
                        (exp(-0.5*point_distance/(0.05*0.05))*          // R(x,x')
                        fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                        fe_values.shape_value(j, l_index) * // grad phi_j(x_l)
                        fe_values.JxW(q_index)*fe_values.JxW(l_index));           // dx
                }
            }
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_stiffness_matrix,
                                         local_dof_indices,
                                         system_stiffness_matrix);

          constraints.distribute_local_to_global(cell_mass_matrix,
                                         local_dof_indices,
                                         system_mass_matrix);
       }
    }

system_mass_matrix.compress(VectorOperation::add);
system_stiffness_matrix.compress(VectorOperation::add);

//system_mass_matrix.print(std::cout);
//system_stiffness_matrix.print(std::cout);
}

void ParallelKL::solve()
{
  const unsigned int num_eigenpairs_requested = 100;

  eigenvalues.resize(num_eigenpairs_requested);
  eigenvectors.resize(num_eigenpairs_requested);

  normalized_gaussian.resize(num_eigenpairs_requested);

  for (unsigned int i = 0; i < num_eigenpairs_requested; ++i)
    eigenvectors[i].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  SolverControl eigen_solver_control (10000, 1e-10);

  SLEPcWrappers::SolverKrylovSchur eigensolver(eigen_solver_control, mpi_communicator);

  eigensolver.set_which_eigenpairs(EPS_LARGEST_REAL);

  eigensolver.set_problem_type(EPS_GHEP);

  pcout << "Beginning Eigensolve..." << std::endl;
  eigensolver.solve(system_stiffness_matrix,
                    system_mass_matrix,
                    eigenvalues,
                    eigenvectors,
                    num_eigenpairs_requested);

  for (unsigned int i = 0; i < num_eigenpairs_requested; i++)
  {
    double temporary_sample = 0.0;

    if (this_mpi_process == 0)
      temporary_sample = distribution(generator);

    double temporary_sum = 0.0;

    MPI_Allreduce (&temporary_sample, &temporary_sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    normalized_gaussian[i] = temporary_sum;

  }
}

void ParallelKL::create_random_field()
{
  randomfield_vector = 0.0;

  PETScWrappers::MPI::Vector tmp_locally_owned_vector(locally_owned_dofs, mpi_communicator);

  for (unsigned int i = 0; i < eigenvalues.size(); i++)
    {
      const double multiplier = sqrt(eigenvalues[i])*normalized_gaussian[i];
      tmp_locally_owned_vector = eigenvectors[i];
      randomfield_vector.add(multiplier, tmp_locally_owned_vector);
    }
}

void ParallelKL::stats_for_random_field()
{
  PETScWrappers::MPI::Vector relevent_randomfield_vector(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  relevent_randomfield_vector = randomfield_vector;

  QGauss<2> quadrature_formula(fe.degree + 1);

  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_JxW_values);

  double vol_on_current_processor = 0.;
  double random_field_volume_integral_on_current_processor = 0.;

  std::vector<double> local_dof_values(fe.dofs_per_cell);
  std::vector<double> current_function_values(quadrature_formula.size());

  const FEValuesExtractors::Scalar r_field(0);

  for (const auto &cell : dof_handler.active_cell_iterators())
  { if (cell->is_locally_owned())
       {
          fe_values.reinit(cell);
          cell->get_dof_values(relevent_randomfield_vector, local_dof_values.begin(), local_dof_values.end());
          fe_values[r_field].get_function_values_from_local_dof_values(local_dof_values, current_function_values);

          for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
              vol_on_current_processor  += fe_values.JxW(q_index);
              random_field_volume_integral_on_current_processor += current_function_values[q_index]*fe_values.JxW(q_index);
            }
       }
  }

  double total_random_field_volume_integral = 0.0;
  double total_volume = 0.0;

  MPI_Allreduce (&random_field_volume_integral_on_current_processor, &total_random_field_volume_integral, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
  MPI_Allreduce (&vol_on_current_processor, &total_volume, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

  double volume_average_of_random_field = total_random_field_volume_integral/total_volume;

  pcout << "Volume Average of Random Field = " << volume_average_of_random_field << std::endl;
  pcout << "Total Volume = " << total_volume << std::endl;

}

void ParallelKL::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  /*
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");
*/
  for (unsigned int i = 0; i < eigenvalues.size(); i++)
  {
    //std::cout << eigenvectors[i] << "\n" << std::endl;
    std::string tmpname = "solution";
    tmpname += Utilities::int_to_string(i, 3);
    //std::cout << "Eigen_val = " << eigenvalues[i] << " " << "Normalized_gaussian = " << normalized_gaussian[i] << "\n"  << std::endl;
    data_out.add_data_vector(eigenvectors[i], tmpname);
  }

  data_out.build_patches();
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}

void ParallelKL::output_results_parallel()
{
    TimerOutput::Scope t(computing_timer, "output parallel");

    PETScWrappers::MPI::Vector relevent_randomfield_vector(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    relevent_randomfield_vector = randomfield_vector;

    DataOut<2> data_out;

    // ############################################################
    // #######                   OUTPUT                     #######
    // ############################################################

    data_out.add_data_vector(dof_handler, relevent_randomfield_vector,
                               "RandomField");

  /*
    for (unsigned int i = 0; i < eigenvalues.size(); i++)
      {
        std::string tmpname = "solution";
        tmpname += Utilities::int_to_string(i, 3);
        data_out.add_data_vector(eigenvectors[i], tmpname);
      }
      */

    std::vector<Vector<double>> buckling_eigenmodes_out;

    buckling_eigenmodes_out.resize(eigenvalues.size());
    for (unsigned int i = 0; i < eigenvalues.size(); ++i)
      {
        buckling_eigenmodes_out[i].reinit(dof_handler.n_dofs(), 0.0);
        buckling_eigenmodes_out[i] = eigenvectors[i];
        const std::string buckling_mode_string = std::string("EigenVector") + Utilities::int_to_string(i, 2);

        data_out.add_data_vector(dof_handler, buckling_eigenmodes_out[i],
                                 buckling_mode_string);
      }

    // ############################################################
    // #######                 WRITE VTU FILE               #######
    // ############################################################
    data_out.build_patches ();
    const std::string filename = "Solution." +
                                  Utilities::int_to_string (this_mpi_process, 2) + ".vtu";

    std::ofstream output (filename.c_str());
    data_out.write_vtu (output);
    output.close();

    // ############################################################
    // #######                 WRITE PVTU FILE              #######
    // ############################################################
    if (this_mpi_process == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i=0; i < n_mpi_processes; ++i)
        filenames.push_back ("Solution." +
                                  Utilities::int_to_string (i, 2) + ".vtu");

      std::ofstream master_output ("Solution.pvtu");

      data_out.write_pvtu_record (master_output, filenames);
    }
}

void ParallelKL::run()
{
  pcout << "Running with " << "PETSc" << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

  make_grid();

  setup_system();

  pcout << "   Number of active cells:       "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  assemble_system();
  solve();
  create_random_field();
  stats_for_random_field();

  if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
    {
      TimerOutput::Scope t(computing_timer, "output");
      //output_results();
      output_results_parallel();

    }
}

int main(int argc, char* argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.depth_console(0);

  ParallelKL klexpansion_2d;
  klexpansion_2d.run();
  return 0;
}
