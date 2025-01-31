#include "boundary.hpp"
#include "spatial.hpp"
#include <cstdint>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

#define tol 1e-3
// #define _debug

__host__ static inline bool compareContainers(std::vector<float> new_container, std::vector<float> boundary) {
  for (int i = 0; i < 3; ++i) {
    if (new_container[i] != boundary[i]) return false;
  }
  return true; 
} 

/*
   Updates the bounds given a new container vector. This stalls the program until
   it can accurately update the spatial hashmap for a new size. 
   */
__host__ void updateBounds(spatialLookupTable *d_lookup_, sphParticle *d_particles_, std::vector<float> new_container, uint32_t particle_recount, const float h) {
  // Static local var
  static std::vector<float> boundary(3, 0);
  static uint32_t n_particles = 0;
  static uint32_t n_partitions = 0;

  // local vars 
  uint32_t paddedSize = findSquare(particle_recount);
  bool anyChange = false;
  uint32_t copySize = 0;

  // Init static boundary 
  if (boundary[0] == 0) {
    boundary = new_container;
  };

  // Init static particle count
  if (n_particles == 0) { 
    n_particles = particle_recount;
  }
  
  // Init static partition count 
  if (n_partitions == 0) {
    uint32_t partition_recount = 1;
      std::vector<uint32_t> containerCount(3, 0);
      for (int i = 0; i < 3; ++i) {
        containerCount[i] = static_cast<uint32_t>(floor(boundary[i]) / h);
        partition_recount *= static_cast<uint32_t>(containerCount[i]);
      }
      n_partitions = partition_recount;
  }

  // If nothing changed return early
  if (compareContainers(new_container, boundary)) return;
  if (n_particles == particle_recount) return;

  // If the container's size changed
  if (new_container != boundary) {
    boundary = new_container;
    anyChange = true; 

    // Free memory to be resized
    cudaFree(d_lookup_->start_cell);
    cudaFree(d_lookup_->end_cell);
    
    // Find new partition count and container count 
    uint32_t partition_recount = 1;
    std::vector<uint16_t> containerCount;
    for (int i = 0; i < 3; ++i) {
      containerCount[i] = static_cast<uint16_t>(floor(boundary[i]) / h);
      partition_recount *= static_cast<uint32_t>(containerCount[i]);
    }
    
    // Allocate more memory for start and end cell arrays
    uint32_t *new_start, *new_end;
    cudaMalloc(&new_start, partition_recount * sizeof(uint32_t));
    cudaMalloc(&new_end, partition_recount * sizeof(uint32_t));

    // Copy new ptrs into lookup ptrs
    cudaMemcpy(d_lookup_->start_cell, &new_start, sizeof(uint32_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lookup_->end_cell, &new_end, sizeof(uint32_t *), cudaMemcpyHostToDevice);

    // Set new partition count
    n_partitions = partition_recount;
  }

  // If new particle count update particle device pntr 
  if (particle_recount != n_particles) {
    anyChange = true;
    // Set local min and max values from container
    /*float maximum = min(boundary);
    float minimum = maximum - (max(boundary) - min(boundary));*/
    float maximum = 9.0;
    float minimum = 1.0; // Hardcoded 

    // Create new host particle array
    sphParticle *host_particles = (sphParticle *)malloc(particle_recount * sizeof(sphParticle));

    // Set copy size for returning device back to host
    copySize = (particle_recount < n_particles) ? particle_recount : n_particles;

    // Copy Back to host
    cudaMemcpy(host_particles, d_particles_, copySize * sizeof(sphParticle), cudaMemcpyDeviceToHost);
    
    // Free old device ptr
    cudaFree(d_particles_);

    // Create memory for new size
    cudaMalloc(&d_particles_, particle_recount * sizeof(sphParticle));

    // If new size is larger than initialize new particles
    if (particle_recount > n_particles) { 
      for (uint32_t idx = n_particles; idx < particle_recount; ++idx) {
        host_particles[idx] = sphParticle(minimum, maximum);
      }
    }
  
    // Copy new particle array back to memory
    cudaMemcpy(d_particles_, host_particles, particle_recount * sizeof(sphParticle), cudaMemcpyHostToDevice);
  }

  // Adjust table size if necessary
  if (paddedSize != findSquare(n_particles)) {
      cudaFree(d_lookup_->table_); 

      // Allocate memory for new table
      struct tableEntry *table_;
      cudaMalloc(&table_, paddedSize * sizeof(tableEntry));

      // Copy array to resized table
      cudaMemcpy(d_lookup_->table_, &table_, sizeof(tableEntry *), cudaMemcpyHostToDevice);  
  }
  
  n_particles = particle_recount;

  // Fill the table with new values according to changes
  if (anyChange) {
    hostFillTable(d_lookup_, d_particles_, n_partitions, n_particles, paddedSize, h);
  }
}

  /*
     Kernel call to naively, individually rectify potential out of bounds behavior for a particle 
     */
  __global__ static void boundaryKernel(const struct Container boundary, sphParticle *d_particles_, uint32_t n_particles, int32_t n_partitions, const float absRadius) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_particles) return;

    const float restitution = 0.8;
    
    // Overlap values
    float upperDistance[3], lowerDistance[3];
#ifdef _debug
    printf("Idx: %u\n", idx);
#endif
    for (int i = 0; i < 3; ++i) {
      // Find potential overlap values
#ifdef _debug
      printf("mass: %f\n", d_particles_[0].mass);
#endif
      upperDistance[i] = boundary.upper[i] - d_particles_[idx].position[i] - absRadius;
      lowerDistance[i] = d_particles_[idx].position[i] + absRadius - boundary.lower[i];
#ifdef _debug  
      printf("Upper: %f\n", upperDistance[i]);
      printf("Lower: %f\n", lowerDistance[i]);
#endif 
      // Rectify overlap 
      if (upperDistance[i] < absRadius + tol) {
        d_particles_[idx].velocity[i] *= -restitution;
        d_particles_[idx].position[i] -= upperDistance[i];

      } else if (lowerDistance[i] < absRadius + tol) {

        d_particles_[idx].velocity[i] *= -restitution; 
        d_particles_[idx].position[i] += lowerDistance[i];

      }
    }
  }

  /*
     Host call to handle the call to boundaryKernel 
     Sets the range of acceptable thread idx to act on particles 
     */
  __host__ void callToBoundaryConditions(struct Container boundary, sphParticle *d_particles_, uint32_t n_particles, uint32_t n_partitions, const float h) {
    uint32_t threadsPerBlock = 256; 
    uint32_t gridSize = (n_particles + threadsPerBlock - 1) / threadsPerBlock; 
#ifdef _debug          
    std::cout << "Test Mass: " << d_particles_[0].mass << '\n'; // This should work since it hasn't been migrated to the cpu yet
#endif
    // Iterate over each container and call kernel for each one 
    boundaryKernel<<<gridSize, threadsPerBlock>>>(
      boundary, d_particles_, n_particles, n_partitions, h * 0.2
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
