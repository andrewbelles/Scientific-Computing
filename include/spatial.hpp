#ifndef __SPATIAL_HPP__
#define __SPATIAL_HPP__

// Cuda headers
#include <cstdio>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// std headers
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstdint>

#define tol 1e-5

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
struct Lookup {
  struct tableEntry *table_;
  uint32_t *start_cell;
  uint32_t *end_cell;
}; 

/*
 * Refactor: 
 * Update particle class to better coalesce memory
 * Only one particle class will be constructed
 * Each member will be initialized as malloc managed, filled, and sent to GPU
 * The class will store the specific device ptr instead of individual data values per struct 
 *
 * In general this is a conversion from an array of structures to a structure of arrays 
 * to better match GPU memory access
 */

class particleContainer {
 public:
  float *positions;
  float *velocities;
  float *pressure_forces;
  float *viscosity_forces; // 3D flattened to 1D of size n_particles * 3
  float *masses;
  float *densities;
  float *pressures; // Flat arrays of size n_particles

  // Base constructor (if resized)
  particleContainer() {
    positions         = nullptr;
    velocities       = nullptr;
    pressure_forces  = nullptr;
    viscosity_forces = nullptr;
    masses           = nullptr;
    densities        = nullptr;
    pressures        = nullptr;
  }

  // Default constructor to create ptrs on host to be transfered to device 
  // Assumes each individual ptr has already been allocated as mallocmanaged
  particleContainer(
    float *u_pos, 
    float *u_vel, 
    float *u_prf,
    float *u_visf,
    float *u_mass,
    float *u_dens,
    float *u_pr,  // Unified ptrs to be set in value
    uint32_t n_particles,
    float min,
    float max
  ) {
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    gen = std::mt19937(seed);
    pos_dis = std::uniform_real_distribution<float>(min, max);
    vel_dis = std::uniform_real_distribution<float>(-max / 20.0, max / 20.0);

    for (uint32_t i = 0; i < n_particles; ++i) {
      // Handle 3D flattened vectors 
      for (int j = 0; j < 3; ++j) {
          uint32_t idx = n_particles * j + i;
          // Set vals
          u_pos[idx]  = pos_dis(gen);
          u_vel[idx]  = vel_dis(gen);
          u_prf[idx]  = 0.0;
          u_visf[idx] = 0.0;
      }

      // Handle 1D vectors
      u_mass[i] = 0.954;
      u_dens[i] = tol;
      u_pr[i]   = 0.0;
    }
    
    // Set ptrs in class
    positions        = u_pos;
    velocities       = u_vel;
    pressure_forces  = u_prf;
    viscosity_forces = u_visf;
    masses           = u_mass;
    densities        = u_dens;
    pressures        = u_pr;
  }

  // Add new position and velocities for new particles if resized
  void addNewParticles(
    float *n_pos,
    float *n_vel,
    uint32_t oldSize,
    uint32_t newSize,
    float min,
    float max
  ) { 
    // Create new position and velocity values
    for (uint32_t idx = oldSize; idx < newSize; ++idx) {
      for (int j = 0; j < 3; ++j) {
        uint32_t co = j * newSize + idx;
        n_pos[co] = pos_dis(gen);
        n_vel[co] = vel_dis(gen);
      }
    }
  }

  // Set accumulator values to base if array was resized 
  void slowSetAccumulators(
    float *u_prf,
    float *u_visf,
    float *u_mass,
    float *u_dens,
    float *u_pr,
    uint32_t n_particles
  ) {
    for (uint32_t i = 0; i < n_particles; ++i) {
      
      for (int j = 0; j < 3; ++j) {
        uint32_t idx = j * n_particles + i;
        u_prf[idx]  = 0.0;
        u_visf[idx] = 0.0;
      }

      u_mass[i] = tol;
      u_dens[i] = tol;
      u_pr[i]   = 0.0;
    }

    // Set accumulators
    pressure_forces = u_prf;
    viscosity_forces = u_visf;
    masses = u_mass;
    densities = u_dens;
    pressures = u_pr;
  }

  // Default destructor
  ~particleContainer() {
    cudaFree(positions);
    cudaFree(velocities);
    cudaFree(pressure_forces);
    cudaFree(viscosity_forces);
    cudaFree(masses);
    cudaFree(densities);
    cudaFree(pressures);
  }
 private:
  std::mt19937 gen;
  std::uniform_real_distribution<float> pos_dis, vel_dis; 
};

/* Function prototypes */

__host__ void setGridSize(uint32_t *blocks, uint32_t *threads, uint32_t arr_size);

template <typename T>
bool isPrime(T val);

template <typename T>
void convertToPrime(T *val);

template <typename T>
T min(const std::vector<T>& vec);

template <typename T>
T max(const std::vector<T>& vec);

template <typename T>
T findSquare(T value);

__device__ int3 positionToCellCoord(float3 position, const float h);
__device__ uint32_t hashPosition(int3 cell_coord, uint32_t n_partition);

__host__ void bitonicSort(Lookup *d_lookup_, uint32_t paddedSize);

__host__ void hostFillTable(
  Lookup *d_lookup_,
  particleContainer *d_particleContainer_,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t paddedSize,
  const float h
);

__host__ void initalizeSimulation(
  Lookup **d_lookup_,
  particleContainer **d_particleContainer_,
  const std::vector<float> container,
  uint32_t *n_partitions,
  uint32_t n_particles,
  const float h
);

#endif // __SPATIAL_HPP__ 
