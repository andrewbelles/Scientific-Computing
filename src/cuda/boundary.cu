#include "boundary.hpp"
#include "spatial.hpp"
#include <cstdint>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

#define tol 1e-4
// #define _debug

__host__ static inline bool equal(std::vector<float> new_container, std::vector<float> boundary) {
  for (int i = 0; i < 3; ++i) {
    if (new_container[i] != boundary[i]) return false;
  }
  return true; 
} 

/*
   Updates the bounds given a new container vector. This stalls the program until
   it can accurately update the spatial hashmap for a new size. 
   */
__host__ void updateBounds(spatialLookupTable *d_lookup_, particleContainer *d_particleContainer_, std::vector<float> new_container, uint32_t particle_recount, const float h) {
  // refac cuda mem manage now pass cpu resize 
  static std::vector<float> boundary; 
  static uint32_t n_particles  = 0;
  static uint32_t n_partitions = 0;

  bool setLookup = false;

  // Set static size variables

  // Set boundary size
  if (boundary.size() == 0) {
    boundary = std::vector<float>(3, 0);
    boundary = new_container;
  }

  // Set partition count
  if (!n_partitions) {
    // Count the number of partitions ("volume")
    uint32_t partition_counter[3];
    n_partitions = 1;
    for (int i = 0; i < 3; ++i) {
      partition_counter[i] = static_cast<uint32_t>(float(boundary[i] / h));
      n_partitions *= partition_counter[i];
    }

  }

  // Set particle count
  if (!n_particles) {
    n_particles = particle_recount;
  } 

  // Resize particle related structs 
  if (particle_recount != n_particles) {

    // Change lookup table size
    if (findSquare(particle_recount) != findSquare(n_particles))
      setLookup = true;
    
    float minimum = 1.0;
    float maximum = 9.0;  // Hardcoded for now

    // We can pull the position and velocity vectors and then delete everything else and create a new ptr
    float *n_pos, *n_vel;
    
    // Create new memory 
    cudaMallocManaged(&n_pos, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_vel, particle_recount * 3 * sizeof(float));
    
    // Set copy size
    uint32_t copy = (n_particles < particle_recount) ? n_particles : particle_recount;

    cudaMemcpy(n_pos, d_particleContainer_->positions, copy * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(n_vel, d_particleContainer_->velocities, copy * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    delete (d_particleContainer_);

    float *n_prf, *n_visf, *n_mass, *n_dens, *n_pr;

    new (d_particleContainer_) particleContainer();

    // Allocate accumulators
    cudaMallocManaged(&n_prf, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_visf, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_mass, particle_recount * sizeof(float));
    cudaMallocManaged(&n_dens, particle_recount * sizeof(float));
    cudaMallocManaged(&n_pr, particle_recount * sizeof(float));

    // Add new particles if size is larger
    if (particle_recount > n_particles) {
      d_particleContainer_->addNewParticles(
        n_pos,
        n_vel,
        n_particles, 
        particle_recount,
        minimum,
        maximum
      );
    }

    d_particleContainer_->slowSetAccumulators(
      n_prf,
      n_visf,
      n_mass,
      n_dens,
      n_pr,
      particle_recount
    );
  }

  // Resize container related structs
  if (!equal(new_container, boundary) || setLookup) {
    // Not worrying about it right now
  }
}
/*
   Kernel call to naively, individually rectify potential out of bounds behavior for a particle 
   */
__global__ static void boundaryKernel(const struct Container boundary, particleContainer *d_particleContainer_, uint32_t n_particles, int32_t n_partitions, const float absRadius) {
uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  const float restitution = 0.8;
  
  // Overlap values
  float upperDistance[3], lowerDistance[3];
#ifdef _debug
  printf("Idx: %u\n", idx);
#endif
  for (int i = 0; i < 3; ++i) {

    uint32_t co = idx + i * n_particles;
    // Find potential overlap values
#ifdef _debug
    printf("mass: %f\n", d_particles_[0].mass);
#endif
    upperDistance[i] = boundary.upper[i] - d_particleContainer_->positions[co] - absRadius;
    lowerDistance[i] = d_particleContainer_->positions[co] + absRadius - boundary.lower[i];
#ifdef _debug  
    printf("Upper: %f\n", upperDistance[i]);
    printf("Lower: %f\n", lowerDistance[i]);
#endif 
    // Rectify overlap 
    if (upperDistance[i] < absRadius + tol) {
      d_particleContainer_->velocities[co] *= -restitution;
      d_particleContainer_->positions[co] -= upperDistance[i];

    } else if (lowerDistance[i] < absRadius + tol) {

      d_particleContainer_->velocities[co] *= -restitution; 
      d_particleContainer_->positions[co] += lowerDistance[i];

    }
  }
}

/*
   Host call to handle the call to boundaryKernel 
   Sets the range of acceptable thread idx to act on particles 
   */
__host__ void callToBoundaryConditions(struct Container boundary, particleContainer *d_particleContainer_, uint32_t n_particles, uint32_t n_partitions, const float h) {
  uint32_t threadsPerBlock = 256; 
  uint32_t gridSize = (n_particles + threadsPerBlock - 1) / threadsPerBlock; 
#ifdef _debug          
  std::cout << "Test Mass: " << d_particles_[0].mass << '\n'; // This should work since it hasn't been migrated to the cpu yet
#endif
  // Iterate over each container and call kernel for each one 
  boundaryKernel<<<gridSize, threadsPerBlock>>>(
    boundary, d_particleContainer_, n_particles, n_partitions, h * 0.2
  );    // First time d_particles_ is called by device -> Must be migrated to device
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    std::cerr << "Bound Launch Error: " << cudaGetErrorString(launchErr) << '\n';
    exit(EXIT_FAILURE);
  }   // Issue isn't a launch error (?) I don't know how that delineates 

  cudaDeviceSynchronize();
  
  cudaError_t boundErr = cudaGetLastError();
  if (boundErr != cudaSuccess) {
    std::cerr << "Bound Error: " << cudaGetErrorString(boundErr) << '\n';
    exit(EXIT_FAILURE);
  }
}
