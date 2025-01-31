#ifndef __SPATIAL_HPP__
#define __SPATIAL_HPP__

// Cuda headers
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// std headers
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdint>

/**
 * Single entry into key/id paired table
 */
struct tableEntry {
  uint32_t cell_key;
  uint32_t idx;
};

/**
 * Spatial Lookup Table with paired start and end cell arrays
 */
struct spatialLookupTable {
  struct tableEntry *table_;
  uint32_t *start_cell;
  uint32_t *end_cell;
}; 

/**
 * Class : getKey, hashFunc(?) 
 * Single sph fluid particle to be placed in GPU memory
 */
class sphParticle {
 public: 
  float position[3];
  float velocity[3];
  float pressure_force[3];
  float viscosity_force[3];
  float mass;
  float density;
  float pressure;

  // Handle vector allocation in GPU later (?)
  sphParticle(float min, float max) : mass(1.0), density(0.0), pressure(0.0) {
    // Seed rng using system time
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> pos_dis(min, max), vel_dis(-max / 20.0, max / 20.0);
    
    // Initialization
    for (int i = 0; i < 3; ++i) {
      velocity[i]        = vel_dis(gen);
      pressure_force[i]  = 0.0;
      viscosity_force[i] = 0.0;
      
      // Randomly distributed
      position[i] = pos_dis(gen);
    }  
  }
};

/* Function prototypes */

template <typename T>
T min(const std::vector<T>& vec);

template <typename T>
T max(const std::vector<T>& vec);

template <typename T>
T findSquare(T value);

__device__ void positionToCellCoord(uint32_t cell_coord[3], const float position[3], const float h);
__device__ uint32_t hashPosition(const uint32_t cell_coord[3], uint32_t n_partition);

__host__ void bitonicSort(spatialLookupTable *d_lookup_, uint32_t paddedSize);

__host__ void hostFillTable(
  spatialLookupTable *d_lookup_,
  sphParticle *d_particles_,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t paddedSize,
  const float h
);

__host__ void initalizeSimulation(
  spatialLookupTable **d_lookup_,
  sphParticle **d_particles_,
  const std::vector<float> container,
  uint32_t *n_partitions,
  uint32_t n_particles,
  const float h
);

#endif // __SPATIAL_HPP__ 
