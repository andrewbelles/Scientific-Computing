#ifndef __INTEGRATE_HPP__
#define __INTEGRATE_HPP__

#include <cuda_runtime.h>
#include "spatial.hpp"

/* Cubic Spline Family of Approximating Kernels */ 
__host__ __device__ float cubicSpline(float distance, float smooth_radius);
__host__ __device__ float gradCubicSpline(float distance, float smooth_radius);
__host__ __device__ float laplacianCubicSpline(float distance, float smooth_radius);

/* Host function to call search kernel and approximate forces for each particle */
__host__ void callToNeighborSearch(
  float *average_neighbor_count,
  spatialLookupTable *d_lookup_,
  particleContainer *d_particleContainer_,
  uint32_t *neighbors,
  uint32_t *neighbor_offset,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t containerCount[3],
  uint32_t list_size, 
  const float h
);

/* Verlet integration passes */
__global__ void firstVerletKernel(particleContainer *d_particleContainer_, uint32_t n_particles);
__global__ void secondVerletKernel(particleContainer *d_particleContainer_, uint32_t n_particles);

__host__ void initOffsetTable();
__host__ void allocateNeighborArrays(
  uint32_t **neighbors,
  uint32_t **neighbor_offset,
  uint32_t n_particles,
  uint32_t *list_size
);

#endif // __INTEGRATE_HPP__
