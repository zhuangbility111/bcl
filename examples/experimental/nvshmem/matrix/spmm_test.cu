// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#define __thrust_compiler_fence() __sync_synchronize()
// #include <cusp/io/matrix_market.h>
// #include <cusp/csr_matrix.h>
// #include <cusp/array2d.h>
// #include <cusp/multiply.h>
// #include <cusp/array2d.h>
// #include <cusp/print.h>

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <thrust/sort.h>

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>
#include <bcl/containers/experimental/cuda/algorithms/algorithm.hpp>

#include <unordered_map>

#include <chrono>
// #include <essl.h>

#include "util.hpp"

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init();

  using T = float;
  using index_type = int;

  bool verify_result = false;

  std::string fname = std::string(argv[1]);

  // Number of vecs in SpMM (width of multi-vec, matrix)
  size_t num_vecs = std::atoi(argv[2]);

  // get matrix shape by reading the file header (meta data)
  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t k = matrix_shape.shape[1];
  size_t n = num_vecs;

  // get the tile shape for each processor based on the matrix shape
  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(m, n, k);

  using allocator_type = BCL::cuda::bcl_allocator<T>;
  using indexing_type = BCL::cuda::RowMajorIndexing;

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T, index_type> a(fname, std::move(blocks[0]));
  BCL::print("Reading B...\n");
  BCL::cuda::Matrix<T, indexing_type> b(k, n, std::move(blocks[1]));
  BCL::cuda::Matrix<T, indexing_type> c(m, n, std::move(blocks[2]));
  b = 1;
  c = 0;
  BCL::cuda::barrier();


  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
    printf("C:\n");
    c.print_info();
  }

  cusparseStatus_t status = cusparseCreate(&BCL::cuda::bcl_cusparse_handle_);

  // printf("A taking %lf GB, B %lf GB\n", 1.0e-9*a.my_mem(), 1.0e-9*b.my_mem());

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::gemm(a, b, c);
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double max_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::max<double>{});
  double max_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::max<double>{});
  double max_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::max<double>{});
  double max_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::max<double>{});
  double max_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::max<double>{});

  double min_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::min<double>{});
  double min_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::min<double>{});
  double min_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::min<double>{});
  double min_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::min<double>{});
  double min_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::min<double>{});

  BCL::cuda::duration_issue = BCL::allreduce(BCL::cuda::duration_issue, std::plus<double>{});
  BCL::cuda::duration_sync = BCL::allreduce(BCL::cuda::duration_sync, std::plus<double>{});
  BCL::cuda::duration_compute = BCL::allreduce(BCL::cuda::duration_compute, std::plus<double>{});
  BCL::cuda::duration_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, std::plus<double>{});
  BCL::cuda::duration_barrier = BCL::allreduce(BCL::cuda::duration_barrier, std::plus<double>{});

  BCL::print("SpMM took %lf s\n", duration);

  if (BCL::rank() == 0) {
    printf("duration_issue %lf (%lf -> %lf)\n",
           BCL::cuda::duration_issue / BCL::nprocs(),
           min_issue, max_issue);
    printf("duration_sync %lf (%lf -> %lf)\n",
           BCL::cuda::duration_sync / BCL::nprocs(),
           min_sync, max_sync);
    printf("duration_compute %lf (%lf -> %lf)\n",
           BCL::cuda::duration_compute / BCL::nprocs(),
           min_compute, max_compute);
    printf("duration_accumulate %lf (%lf -> %lf)\n",
           BCL::cuda::duration_accumulate / BCL::nprocs(),
           min_accumulate, max_accumulate);
    printf("duration_barrier %lf (%lf -> %lf)\n",
           BCL::cuda::duration_barrier / BCL::nprocs(),
           min_barrier, max_barrier);
  }

  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  if (BCL::rank() == 0 && verify_result) {
    fprintf(stderr, "Reading in matrix...\n");
    BCL::CSRMatrix<T, index_type> mat(fname);
    fprintf(stderr, "Copying to GPU...\n");
    auto local_a = BCL::cuda::to_gpu<T, index_type, allocator_type>(mat);

    fprintf(stderr, "Creating local b...\n");
    BCL::cuda::CudaMatrix<T, allocator_type, indexing_type> local_b({k, n});
    fprintf(stderr, "Creating local c...\n");
    BCL::cuda::CudaMatrix<T, allocator_type, indexing_type> local_c({m, n});

    fprintf(stderr, "Writing to matrices...\n");
    local_b = 1;
    local_c = 0;

    fprintf(stderr, "Doing local spmm...\n");
    BCL::cuda::spmm_cusparse(local_a, local_b, local_c);
    cudaDeviceSynchronize();

    fprintf(stderr, "Getting C matrix...\n");
    auto distributed_c = c.get_matrix();

    std::vector<T> local_data(local_c.size());
    cudaMemcpy(local_data.data(), local_c.data(), sizeof(T)*local_c.size(), cudaMemcpyDeviceToHost);

    assert(distributed_c.size() == local_c.size());
    fprintf(stderr, "Checking accuracy...\n");
    T eps = 1.0e-5;
    size_t matching = 0;
    bool print = false;
    for (size_t i = 0; i < c.shape()[0]; i++) {
      for (size_t j = 0; j < c.shape()[1]; j++) {
        size_t d_idx = i*c.shape()[1] + j;
        size_t l_idx = indexing_type().index(i, j, local_c.ld());
        // if (std::abs(distributed_c[d_idx] - local_data[l_idx]) > eps) {
        if (!is_equal(T(distributed_c[d_idx]), T(local_data[l_idx]))) {
          // assert(false);
          if (print) {
            printf("O %2.2lf != %2.2lf ", distributed_c[d_idx], local_data[l_idx]);
          }
        } else {
          if (print) {
            printf("X %2.2lf == %2.2lf ", distributed_c[d_idx], local_data[l_idx]);
          }
          matching++;
        }
      }
      if (print) {
        printf("\n");
      }
    }
    /*
    for (size_t i = 0; i < distributed_c.size(); i++) {
      if (std::abs(distributed_c[i] - local_data[i]) > eps) {
        // fprintf(stderr, "[%lu] %f != %f\n", i, distributed_c[i], local_data[i]);
      } else {
        matching++;
      }
    }
    */
    printf("%lu / %lu (%lf%%) indices match.\n", matching, distributed_c.size(),
           100 * ((double) matching) / distributed_c.size());
    if (matching == distributed_c.size()) {
      printf("OK.\n");
    } else {
      printf("***FAILED!***\n");
    }
  }

  // Benchmark function to measure performance
  auto benchmark_gemm = [&](int warmup_rounds, int run_rounds) {
    BCL::print("Starting benchmark with %d warmup rounds and %d run rounds...\n", warmup_rounds, run_rounds);
    
    // Reset timing variables
    BCL::cuda::duration_issue = 0.0;
    BCL::cuda::duration_sync = 0.0;
    BCL::cuda::duration_compute = 0.0;
    BCL::cuda::duration_accumulate = 0.0;
    BCL::cuda::duration_barrier = 0.0;
    
    // Warmup rounds
    BCL::print("Warmup phase...\n");
    for (int i = 0; i < warmup_rounds; i++) {
      c = 0;  // Reset result matrix
      BCL::cuda::barrier();
      BCL::cuda::gemm(a, b, c);
      BCL::cuda::barrier();
    }
    
    // Reset timing variables after warmup
    BCL::cuda::duration_issue = 0.0;
    BCL::cuda::duration_sync = 0.0;
    BCL::cuda::duration_compute = 0.0;
    BCL::cuda::duration_accumulate = 0.0;
    BCL::cuda::duration_barrier = 0.0;
    
    // Benchmark runs
    BCL::print("Benchmark phase...\n");
    std::vector<double> run_times;
    
    for (int i = 0; i < run_rounds; i++) {
      c = 0;  // Reset result matrix
      BCL::cuda::barrier();
      
      auto run_begin = std::chrono::high_resolution_clock::now();
      BCL::cuda::gemm(a, b, c);
      BCL::cuda::barrier();
      auto run_end = std::chrono::high_resolution_clock::now();
      
      double run_duration = std::chrono::duration<double>(run_end - run_begin).count();
      run_times.push_back(run_duration);
      
      if (BCL::rank() == 0) {
        printf("Run %d: %lf s\n", i + 1, run_duration);
      }
    }
    
    // Calculate statistics
    double total_time = 0.0;
    double min_time = run_times[0];
    double max_time = run_times[0];
    
    for (double time : run_times) {
      total_time += time;
      min_time = std::min(min_time, time);
      max_time = std::max(max_time, time);
    }
    
    double avg_time = total_time / run_rounds;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double time : run_times) {
      variance += (time - avg_time) * (time - avg_time);
    }
    variance /= run_rounds;
    double std_dev = sqrt(variance);
    
    // Aggregate timing statistics across all processes
    double avg_issue = BCL::cuda::duration_issue / run_rounds;
    double avg_sync = BCL::cuda::duration_sync / run_rounds;
    double avg_compute = BCL::cuda::duration_compute / run_rounds;
    double avg_accumulate = BCL::cuda::duration_accumulate / run_rounds;
    double avg_barrier = BCL::cuda::duration_barrier / run_rounds;
    
    // Reduce to get global statistics
    double global_avg_time = BCL::allreduce(avg_time, std::plus<double>{}) / BCL::nprocs();
    double global_min_time = BCL::allreduce(min_time, BCL::min<double>{});
    double global_max_time = BCL::allreduce(max_time, BCL::max<double>{});
    double global_std_dev = BCL::allreduce(std_dev, std::plus<double>{}) / BCL::nprocs();
    
    double global_avg_issue = BCL::allreduce(avg_issue, std::plus<double>{}) / BCL::nprocs();
    double global_avg_sync = BCL::allreduce(avg_sync, std::plus<double>{}) / BCL::nprocs();
    double global_avg_compute = BCL::allreduce(avg_compute, std::plus<double>{}) / BCL::nprocs();
    double global_avg_accumulate = BCL::allreduce(avg_accumulate, std::plus<double>{}) / BCL::nprocs();
    double global_avg_barrier = BCL::allreduce(avg_barrier, std::plus<double>{}) / BCL::nprocs();
    
    // Print benchmark results
    if (BCL::rank() == 0) {
      printf("\n=== Benchmark Results ===\n");
      printf("Warmup rounds: %d\n", warmup_rounds);
      printf("Benchmark runs: %d\n", run_rounds);
      printf("Average time: %lf s (std dev: %lf s)\n", global_avg_time, global_std_dev);
      printf("Min time: %lf s\n", global_min_time);
      printf("Max time: %lf s\n", global_max_time);
      printf("\nDetailed timing breakdown (averages):\n");
      printf("  Issue time: %lf s\n", global_avg_issue);
      printf("  Sync time: %lf s\n", global_avg_sync);
      printf("  Compute time: %lf s\n", global_avg_compute);
      printf("  Accumulate time: %lf s\n", global_avg_accumulate);
      printf("  Barrier time: %lf s\n", global_avg_barrier);
      printf("========================\n");
    }
    
    return global_avg_time;
  };
  
  // Run benchmark with 5 warmup rounds and 10 measurement rounds
  double benchmark_time = benchmark_gemm(5, 10);
  
  BCL::print("Benchmark completed. Average performance: %lf s\n", benchmark_time);

  BCL::finalize();
  return 0;
}
