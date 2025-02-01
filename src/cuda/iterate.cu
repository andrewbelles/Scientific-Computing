#include "spatial.hpp"
#include "boundary.hpp"
#include "integrate.hpp"
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define _errorcheck
#define substep 1
#define tol 1e-4
// #define _debug
// #define _verbose

// __global__ static void printPositions(sphParticle *d_particles_, uint32_t n_particles) {  
//  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
//  if (idx >= n_particles) return;

#ifdef _verbose
    printf("idx: %u : <%f,%f,%f>\n",
      idx, 
      d_particles_[idx].position[0],
      d_particles_[idx].position[1],
      d_particles_[idx].position[2]
    );
#endif
// }

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
#ifdef _debug  
    printf("Position %d for idx: %u : %f\n",
      i, idx, d_positions[idx * 3 + i]
    );
#endif
  }
}

/* Generates positions from particle array */
__host__ void particleIterator(
  spatialLookupTable *d_lookup_,
  particleContainer *d_particleContainer_, 
  float **u_positions,
  float **u_densities,
  std::vector<float> container,
  uint32_t n_particles,
  uint32_t n_partitions,
  const float h
)
{
  uint32_t threadsPerBlock = 256;
  uint32_t gridSize = (n_particles + threadsPerBlock - 1) / threadsPerBlock;
  uint32_t verletGrid = (gridSize < 30) ? 30 : gridSize;
  // threadsPerBlock ~= (n_particles - 1) / (verletGrid - 1) -> findSquare()
  uint32_t verletTPB = findSquare((n_particles - 1) / (verletGrid - 1));

#ifdef _verbose  
  printPositions<<<gridSize, threadsPerBlock>>>(d_particles_, n_particles);
#endif
  // Update bounds if container or particle count have changed
  updateBounds(d_lookup_, d_particleContainer_, container, n_particles, h);

  struct Container boundary = {
    .lower = {0.0, 0.0, 0.0},
    .upper = {container[0], container[1], container[2]}
  };

  setAccumulators<<<gridSize, threadsPerBlock>>>(d_particleContainer_, n_particles);

  // Set lookup spatialLookupTable for new positions 
  uint32_t padded_size = findSquare(n_particles);
  hostFillTable(d_lookup_, d_particleContainer_, n_partitions, n_particles, padded_size, h);

  // Checking boundary conditions is breaking the particles positions...
  
  // Only one static boundary for now (the container itself)
  for (int i = 0; i < substep; ++i) { 
    callToBoundaryConditions(boundary, d_particleContainer_, n_particles, n_partitions, h);
  }
#ifdef _verbose
  printPositions<<<gridSize, threadsPerBlock>>>(d_particles_, n_particles);
#endif
  firstVerletKernel<<<verletGrid, verletTPB>>>(d_particleContainer_, n_particles);
  cudaDeviceSynchronize();
#ifdef _errorcheck
  cudaError_t verletErr = cudaGetLastError();
  if (verletErr != cudaSuccess) {
    std::cerr << "Verlet 1 Error: " << cudaGetErrorString(verletErr) << '\n';
    exit(EXIT_FAILURE);
  }
#endif

  // Convert upper bounds to the max id for each direction
  static uint32_t *containerCount;
  if (containerCount == NULL/* || anyChange == true*/) {
    cudaMallocManaged(&containerCount, 3 * sizeof(uint32_t));
    for (int i = 0; i < 3; ++i) {
      containerCount[i] = static_cast<uint32_t>(floor(container[i] / h));
    }
  };

#ifdef _debug
  for (int i = 0; i < 3; ++i) {
    std::cout << i << ": " << containerCount[i] << '\n';
  }
#endif
  // Calls the neighbor search
  callToNeighborSearch(
    d_lookup_,
    d_particleContainer_,
    n_partitions,
    n_particles,
    containerCount,
    h
  );

  // Second verlet pass with new force values
  secondVerletKernel<<<verletGrid, verletTPB>>>(d_particleContainer_, n_particles);
#ifdef _errorcheck
  verletErr = cudaGetLastError();
  if (verletErr != cudaSuccess) {
    std::cerr << "Verlet 2 Launch Error: " << cudaGetErrorString(verletErr) << '\n';
    exit(EXIT_FAILURE);
  }
#endif

  cudaDeviceSynchronize();
#ifdef _errorcheck
  verletErr = cudaGetLastError();
  if (verletErr != cudaSuccess) {
    std::cerr << "Verlet 2 Sync Error: " << cudaGetErrorString(verletErr) << '\n';
    exit(EXIT_FAILURE);
  }
#endif
#ifdef _debug
  std::cout << "Completed Position Set\n";
#endif
  // If first iteration create managed malloc call
  if ((*u_positions) == NULL) {
    cudaMallocManaged(u_positions, n_particles * sizeof(float) * 3);
  }

  if ((*u_densities) == NULL) {
    cudaMallocManaged(u_densities, n_particles * sizeof(float));
  }

  // Creates single contiguous bufr of floats (for positions)
  updateHostBuffer<<<gridSize, threadsPerBlock>>>(d_particleContainer_, (*u_positions), (*u_densities), n_particles);

#ifdef _errorcheck 
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    std::cerr << "Launch Error: " << cudaGetErrorString(launchErr) << '\n';
    exit(EXIT_FAILURE);
  }
#endif

  cudaDeviceSynchronize();
#ifdef _errorcheck
  cudaError_t syncErr = cudaGetLastError();
  if (syncErr != cudaSuccess) {
    std::cerr << "Sync Error: " << cudaGetErrorString(syncErr) << '\n';
    exit(EXIT_FAILURE);
  }
#endif

#ifdef _debug
  std::cout << "Position Buffer created\n";
#endif

#ifdef _debug
  // fill 2d vector of positions
  for (uint32_t i = 0; i < n_particles; ++i) {
    for (int j = 0; j < 3; ++j) {
      host_positions[i][j] = host_vec[i * 3 + j];
      std::cout << "Position[i][j]: " << host_positions[i][j] << '\n';
    }
  }
#endif
}
