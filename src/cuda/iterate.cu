#include "iterate.hpp"
#include "integrate.hpp"

#define tol 1e-4

__global__ static void setAccumulators(particleContainer *d_objs_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  // Quickly reset all acculated values from previous iteration
  d_objs_->densities[idx]  = tol;
  d_objs_->pressures[idx]  = 0.0;
  for (int i = 0; i < 3; ++i) {

    uint32_t co = idx + i * n_particles;

    d_objs_->pressure_forces[co]  = 0.0;
    d_objs_->viscosity_forces[co] = 0.0;
  }
}

/* Copies positions into contiguous device buffer */ 
__global__ static void updateHostBuffer(
  particleContainer *d_objs_,
  float *u_positions,
  float *u_densities,
  uint32_t n_particles
)
{
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  u_densities[idx] = d_objs_->densities[idx];
  for (int i = 0; i < 3; ++i) {
    u_positions[idx * 3 + i] = d_objs_->positions[idx + i * n_particles];
  }
}

/* Generates positions from particle array */

__host__ void particleIterator(
  neighborList *list,
  particleContainer *d_objs_,
  uint32_t *list_size,
  float **u_positions,
  float **u_densities,
  Lookup *d_lookup_,
  std::vector<float> container,
  uint32_t n_particles,
  uint32_t n_partitions, 
  float h
) {
  static uint32_t blocks = 0, threads = 0;
  cudaError_t err;
  setGridSize(&blocks, &threads, n_particles);
#ifdef __debug
  std::cout << "Grid Set\n";
#endif

  // Update bounds if container or particle count have changed
  updateBounds(d_lookup_, d_objs_, container, n_particles, h);
#ifdef __debug
  std::cout << "Updated Bounds\n";  
#endif
  struct Container boundary = {
    .lower = {0.0, 0.0, 0.0},
    .upper = {container[0], container[1], container[2]}
  };

  setAccumulators<<<blocks, threads>>>(d_objs_, n_particles);
#ifdef __debug
  std::cout << "Set Accumulators\n";
#endif
  // Set lookup Lookup for new positions 
  uint32_t padded_size = findSquare(n_particles);
  hostFillTable(d_lookup_, d_objs_, n_partitions, n_particles, padded_size, h);
#ifdef __debug
  std::cout << "Filled Table\n";
#endif
  // Checking boundary conditions is breaking the particles positions...
  
  // Only one static boundary for now (the container itself)
  callToBoundaryConditions(boundary, d_objs_, n_particles, n_partitions, h);
#ifdef __debug 
  std::cout << "Enforced Boundary\n";
#endif
  // Launch the first half of verlet integration that doesn't require the next step of forces
  firstVerletKernel<<<blocks, threads>>>(d_objs_, n_particles);
  cudaDeviceSynchronize();
#ifdef __debug
  std::cout << "First Verlet Pass\n";
#endif
  // Convert upper bounds to the max id for each direction (static container)
  static uint32_t *containerCount;
  if (containerCount == NULL) {
    cudaMallocManaged(&containerCount, 3 * sizeof(uint32_t));
    for (int i = 0; i < 3; ++i) {
      containerCount[i] = static_cast<uint32_t>(floor(container[i] / h));
    }
  };

  // Calls the neighbor search
  neighborSearch(
    list,
    d_objs_,
    list_size, 
    d_lookup_,
    n_partitions,
    n_particles,
    containerCount,
    h
  );

  // Second verlet pass with new force values
  secondVerletKernel<<<blocks, threads>>>(d_objs_, n_particles);
  cudaDeviceSynchronize();
#ifdef __debug
  std::cout << "Second Verlet Pass\n";
#endif
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
  updateHostBuffer<<<blocks, threads>>>(d_objs_, (*u_positions), (*u_densities), n_particles);

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
#ifdef __debug
  std::cout << "Filled host buffer\n";
#endif
}
