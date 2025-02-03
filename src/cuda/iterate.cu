#include "spatial.hpp"
#include "boundary.hpp"
#include "integrate.hpp"
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define tol 1e-4

__global__ static void setAccumulators(particleContainer *d_particleContainer_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  // Quickly reset all acculated values from previous iteration
  d_particleContainer_->densities[idx]  = tol;
  d_particleContainer_->pressures[idx]  = 0.0;
  for (int i = 0; i < 3; ++i) {

    uint32_t co = idx + i * n_particles;

    d_particleContainer_->pressure_forces[co]  = 0.0;
    d_particleContainer_->viscosity_forces[co] = 0.0;
  }
}

/* Copies positions into contiguous device buffer */ 
__global__ static void updateHostBuffer(
  particleContainer *d_particleContainer_,
  float *u_positions,
  float *u_densities,
  uint32_t n_particles
)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  u_densities[idx] = d_particleContainer_->densities[idx];
  for (int i = 0; i < 3; ++i) {
    u_positions[idx * 3 + i] = d_particleContainer_->positions[idx + i * n_particles];
  }
}

/* Generates positions from particle array */
__host__ void particleIterator(
  float *average_neighbor_count,
  spatialLookupTable *d_lookup_,
  particleContainer *d_particleContainer_, 
  uint32_t *neighbors,
  uint32_t *neighbors_offset,
  uint32_t list_size,
  float **u_positions,
  float **u_densities,
  std::vector<float> container,
  uint32_t n_particles,
  uint32_t n_partitions,
  const float h
)
{
  static uint32_t blocks = 0, threads = 0;
  cudaError_t err;
  setGridSize(&blocks, &threads, n_particles);

  std::cout << "Grid Set\n";

  // Update bounds if container or particle count have changed
  updateBounds(d_lookup_, d_particleContainer_, container, n_particles, h);

  std::cout << "Updated Bounds\n";  

  struct Container boundary = {
    .lower = {0.0, 0.0, 0.0},
    .upper = {container[0], container[1], container[2]}
  };

  setAccumulators<<<blocks, threads>>>(d_particleContainer_, n_particles);

  std::cout << "Set Accumulators\n";

  // Set lookup spatialLookupTable for new positions 
  uint32_t padded_size = findSquare(n_particles);
  hostFillTable(d_lookup_, d_particleContainer_, n_partitions, n_particles, padded_size, h);

  std::cout << "Filled Table\n";

  // Checking boundary conditions is breaking the particles positions...
  
  // Only one static boundary for now (the container itself)
  callToBoundaryConditions(boundary, d_particleContainer_, n_particles, n_partitions, h);
  
  std::cout << "Enforced Boundary\n";

  // Launch the first half of verlet integration that doesn't require the next step of forces
  firstVerletKernel<<<blocks, threads>>>(d_particleContainer_, n_particles);
  cudaDeviceSynchronize();

  std::cout << "First Verlet Pass\n";

  // Convert upper bounds to the max id for each direction (static container)
  static uint32_t *containerCount;
  if (containerCount == NULL) {
    cudaMallocManaged(&containerCount, 3 * sizeof(uint32_t));
    for (int i = 0; i < 3; ++i) {
      containerCount[i] = static_cast<uint32_t>(floor(container[i] / h));
    }
  };

  std::cout << "Call to Neighbor Search\n";

  // Calls the neighbor search
  callToNeighborSearch(
    average_neighbor_count,
    d_lookup_,
    d_particleContainer_,
    neighbors,
    neighbors_offset,
    n_partitions,
    n_particles,
    containerCount,
    list_size,
    h
  );

  std::cout << "Completed Neighbor Search\n";

  // Second verlet pass with new force values
  secondVerletKernel<<<blocks, threads>>>(d_particleContainer_, n_particles);
  cudaDeviceSynchronize();

  std::cout << "Second Verlet Pass\n";

  // If first iteration create managed malloc calls for cpu copy of positions and densities
  if ((*u_positions) == NULL) {
    cudaMallocManaged(u_positions, n_particles * sizeof(float) * 3);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Position Malloc Error: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }
  }

  if ((*u_densities) == NULL) {
    cudaMallocManaged(u_densities, n_particles * sizeof(float));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Density Malloc Error: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }
  }

  // Creates single contiguous buffer of positions and densities on cpu 
  updateHostBuffer<<<blocks, threads>>>(d_particleContainer_, (*u_positions), (*u_densities), n_particles);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Launch Error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Sync Error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  std::cout << "Filled host buffer\n";
}
