#include "integrate.hpp"
#include "spatial.hpp"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <driver_types.h>
#include <vector_functions.h>

// Defines 
#define k 3000
#define rho0 1000
#define dt 1e-3       // Change to dynamically shift in value 
#define viscosity 1e-2

// Offset table for kernel 
__constant__ int3 offset_table[27];

__device__ int global_offset; 

__device__ static inline float magnitude(const float3 a) {
  return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

/* float3 overloads */

__host__ __device__ static inline float3 operator/(const float3 a, const float val) {
  return make_float3(a.x / val, a.y / val, a.z / val);
}

__host__ __device__ static inline float3 operator*(const float3 a, const float val) {
  return make_float3(a.x * val, a.y * val, a.z * val);
}

__host__ __device__ static inline void operator+=(float3 &a, float3 b) {
  float *vec_a[] = {&a.x, &a.y, &a.z};
  float *vec_b[] = {&b.x, &b.y, &b.z};

  for (int i = 0; i < 3; ++i)
    *vec_a[i] += *vec_b[i];
}

__host__ __device__ static inline float3 operator-(const float3 a, const float3 b) {
  return make_float3(a.x - b.z, a.y - b.z, a.z - b.z);
}

__host__ __device__ static inline void operator-=(float3 &a, float3 b) {
  float *vec_a[] = {&a.x, &a.y, &a.z};
  float *vec_b[] = {&b.x, &b.y, &b.z};

  for (int i = 0; i < 3; ++i)
    *vec_a[i] -= *vec_b[i];
}

/*
   Cubic Spline smooth field approximating kernel. 
   */
__host__ __device__ float cubicSpline(float distance, float smooth_radius) {
  // Constant values
  const float  q = distance / smooth_radius;
  const float  a3 = 1.0 / (M_PI * smooth_radius * smooth_radius * smooth_radius);
  float value = a3;

  // Calcuate value of kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (1.0 - (1.5 * q * q) + 0.75 * q * q * q);
  } else if (q >= 1 && q < 2) {
    value *= (0.25 * (2.0 - q) * (2.0 - q) * (2.0 - q)); 
  // Outside influence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/*
   The gradient of the Cubic Spline kernel 
   */
__host__ __device__ float gradCubicSpline(float distance, float smooth_radius) {
  // Constant values
  const float  q = distance / smooth_radius;
  const float  a3 = 1.0 / (M_PI * smooth_radius * smooth_radius * smooth_radius);
  float value = a3;

  // Calculate the gradient of the kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (-3.0 * q + 2.25 * q * q);
  } else if (q >= 1 && q < 2) {
    value *= (-0.75 * (2.0 - q) * (2.0 - q));
  // Outside influence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/*
   The laplacian of the Cubic Spline kernel 
   */
__host__ __device__ float laplacianCubicSpline(float distance, float smooth_radius) {
  const float  q = distance / smooth_radius;
  const float  a3 = 1.0 / (M_PI * smooth_radius * smooth_radius * smooth_radius);
  float value = a3;

  // Calculate the laplacian of the kernel over the smoothing radius
  if (q >= 0 && q < 1) {
    value *= (-3.0 + 4.5 * q);
  } else if (q >= 1 && q < 2) {
    value *= (1.5 * (2.0 - q));
  // Outside incluence
  } else if (q >= 2) {
    value = 0;
  }
  return value;
}

/**
 * Create the offset table and copy symbol to GPU memory 
 */ 
__host__ void initOffsetTable() {
  int3 host_offset[27];

  int idx = 0;
  // Iterate over 3x3x3 grid 
  for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx, ++idx)
        host_offset[idx] = make_int3(dx, dy, dz);  

  // Copy the host table to the global constant offset table
  cudaMemcpyToSymbol(offset_table, host_offset, sizeof(host_offset));
}

__device__ static inline bool inBounds(const int3 a, const uint32_t bounds[3]) {
  if (a.x >= bounds[0] || a.x < 0 || a.y >= bounds[1] || a.y < 0 || a.z >= bounds[2] || a.z < 0) return false;
  return true;
}

/**
 * Operator for int3 adding
 */
__device__ static inline int3 operator+(int3 a, int3 b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__global__ static void computeNeighborList(
  spatialLookupTable *d_lookup_,
  particleContainer *d_particleContainer_,
  uint32_t *neighbors,
  uint32_t *neighbor_offset,
  int *neighbor_count,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t containerCount[3],
  uint32_t list_size,
  int *status,
  const float h
) {
  extern __shared__ uint32_t shared_cell[];

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= n_particles) return;

  // Fetch particle id associated with table
  uint32_t pid = d_lookup_->table_[idx].idx;
  uint32_t start = 0, end = 0, hash = 0, rel = 0, idj = 0;
  float3 displace, relative_pos, local_pos;
  float distance = 0.0;
  int3 relative_coord, cell_coord;
  int count = 0, offset = 0, block_end = 0;

  // Convert local position to float3 type 
  local_pos = make_float3(
    d_particleContainer_->positions[pid],
    d_particleContainer_->positions[pid + 1 * n_particles],
    d_particleContainer_->positions[pid + 2 * n_particles]
  );

  // Collect the cell coordinate
  cell_coord = positionToCellCoord(
    local_pos,
    h
  );

  // Take two passes; once to calculate the offset for each idx then again to actually fill neighbors array 
  for (int pass = 0; pass < 2; ++pass) {
    count = 0;
    for (int i = 0; i < 27; ++i) {
      // Find offset position from table
      relative_coord = cell_coord + offset_table[i];

      if (!inBounds(relative_coord, containerCount)) continue;

      // Find hash of offset position and subsequent rel ids
      hash = hashPosition(relative_coord, n_partitions);
      start = d_lookup_->start_cell[hash];
      end   = d_lookup_->end_cell[hash];

      for (uint32_t j = start; j < end; j += blockDim.x) {
        idj = j + threadIdx.x;  
        if (idj < end) shared_cell[threadIdx.x] = d_lookup_->table_[idj].idx;
        __syncthreads();

        block_end = (end - j < blockDim.x) ? end - j : blockDim.x;

        for (int t = 0; t < block_end; ++t) {
          rel = shared_cell[t];

          if (rel == pid) continue;

          relative_pos = make_float3( 
            d_particleContainer_->positions[rel],
            d_particleContainer_->positions[rel + 1 * n_particles],
            d_particleContainer_->positions[rel + 2 * n_particles]
          );

          displace = local_pos - relative_pos;
          distance = magnitude(displace);

          if (distance > 2.0 * h) continue; 
          
          if (pass == 0) { 
            count++;  // Collect count of neighbors for each idx
          } else {
            if (offset + count >= list_size) {
              atomicExch(status, 1); 
              return; 
            }

            neighbors[offset + count] = rel;
            count++;
          }
        }
        __syncthreads();
      }
    }

    if (pass == 0) {
      offset = atomicAdd(&global_offset, count);
      neighbor_offset[idx] = offset;
    }
  }

  // Increment the offset for the neighbors array 
  atomicAdd(neighbor_count, count);
  printf("Neighbor Count: %d\n", (*neighbor_count));
}

/**
 * Find the density, pressure, and system forces from the built neighbor list
 */
__global__ static void computeForces(
  uint32_t *neighbors,
  uint32_t *neighbor_offset,
  particleContainer *d_particleContainer_,
  uint32_t n_particles,
  const float h
) { 
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 

  // Set local values
  float3 pressure_force  = make_float3(0.0, 0.0, 0.0);
  float3 viscosity_force = make_float3(0.0, 0.0, 0.0);
  float3 local_pos = make_float3(
    d_particleContainer_->positions[idx],
    d_particleContainer_->positions[idx + n_particles],
    d_particleContainer_->positions[idx + 2 * n_particles]
  );
 
  float3 local_vel = make_float3(
    d_particleContainer_->velocities[idx],
    d_particleContainer_->velocities[idx + n_particles],
    d_particleContainer_->velocities[idx + 2 * n_particles]
  );

  // Iterator bounds
  uint32_t start = neighbor_offset[idx];
  uint32_t end   = neighbor_offset[idx + 1];

  // Loop over neighbor indexes
  for (uint32_t i = start; i < end; ++i) {
    // Set the relative particle id
    uint32_t rel = neighbors[i];
    
    float3 relative_pos = make_float3(
      d_particleContainer_->positions[rel],
      d_particleContainer_->positions[rel + n_particles],
      d_particleContainer_->positions[rel + 2 * n_particles] 
    );

    // Find distance
    float3 displace = local_pos - relative_pos;
    float distance  = magnitude(displace);

    float3 relative_vel = make_float3( 
      d_particleContainer_->velocities[idx],
      d_particleContainer_->velocities[idx + n_particles],
      d_particleContainer_->velocities[idx + 2 * n_particles]
    );

    // Find density sum
    d_particleContainer_->densities[idx] += d_particleContainer_->masses[rel] * cubicSpline(distance, h);
    d_particleContainer_->pressures[idx] = k * (d_particleContainer_->densities[idx] - rho0);

    // Unit vector of direction calculatio 
    float3 direction = displace / (distance + tol);
    pressure_force  -= (direction * (d_particleContainer_->pressures[idx] + d_particleContainer_->pressures[rel]) * gradCubicSpline(distance, h));
    viscosity_force += (((local_vel - relative_vel) * d_particleContainer_->masses[rel] / d_particleContainer_->densities[rel]) * laplacianCubicSpline(distance, h));
  }
  float *pressure_force_arr[]  = {&pressure_force.x, &pressure_force.y, &pressure_force.z};
  float *viscosity_force_arr[] = {&viscosity_force.x, &viscosity_force.y, &viscosity_force.z};

  // Set computed force values 
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    d_particleContainer_->pressure_forces[idx + i * n_particles]  = *pressure_force_arr[i];
    d_particleContainer_->viscosity_forces[idx + i * n_particles] = *viscosity_force_arr[i];
  }
}

__global__ void setOffset() {
  global_offset = 0;
}
 
/**
 * Host function to call the search kernel to find each particles forces relative to itself
 */
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
) {
  static uint32_t blocks = 0, threads = 0;
  setGridSize(&blocks, &threads, n_particles);
  int status = 0, neighbor_count = 0;
  cudaError_t err;

  cudaMemPrefetchAsync(neighbors, list_size * sizeof(uint32_t), 0); 
  cudaMemPrefetchAsync(neighbors, (n_particles + 1) * sizeof(uint32_t), 0);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Prefetch: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }

  // Perform computation until neighbor list is valid sized 
  do {
    setOffset<<<1, 1>>>();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Set global offset: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }

    computeNeighborList<<<blocks, threads>>>(
      d_lookup_,
      d_particleContainer_,
      neighbors,
      neighbor_offset,
      &neighbor_count,
      n_partitions,
      n_particles,
      containerCount,
      list_size,
      &status,
      h
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Neighbor List: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }

    // Wait for all threads to complete before restarting if size error
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "List sync: " << cudaGetErrorString(err) << '\n';
      exit(EXIT_FAILURE);
    }
    
    // Neighbors list size error
    if (status == 1) {
      // Take truncated value of 3/2 k and resize neighbors list
      int average_neighbors = list_size / n_particles; 
      average_neighbors *= 0.5;
      cudaFree(neighbors);
      cudaMallocManaged(&neighbors, average_neighbors * n_particles * sizeof(uint32_t));
    }

  } while (status != 0); 

  // Get average number of neighbors per particle for profiling 
  (*average_neighbor_count) = static_cast<float>(neighbor_count) / n_particles;

  // Expected success 
  computeForces<<<blocks, threads>>>(
    neighbors,
    neighbor_offset,
    d_particleContainer_,
    n_particles,
    h
  );
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Force Compute: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }
  
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Force Sync: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }
}

/*
   Completes the first section of verlet integration
   */
__global__ void firstVerletKernel(particleContainer *d_particleContainer_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // Position and velocity update loop
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
   
    // Sums the pressure and viscosity forces for each axis
    forceSum[i] += (d_particleContainer_->pressure_forces[co] + d_particleContainer_->viscosity_forces[co]);
    
    // Integrates the velocity and position
    d_particleContainer_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt));
    d_particleContainer_->positions[co] += (d_particleContainer_->velocities[co] * static_cast<float>(dt));
  }
}

/*
   Second pass of verlet integration
   */
__global__ void secondVerletKernel(particleContainer *d_particleContainer_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // iterate over axis
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
    // Sum forces from previous iteration 
    forceSum[i] += ((d_particleContainer_->pressure_forces[co] + d_particleContainer_->viscosity_forces[co]) / d_particleContainer_->masses[idx]);
    
    // Second half step to fully velocity
    d_particleContainer_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt));
  }
}

/*
  Note: I want to get an idea of an acceptable size for my compressed sparse matrix neighbors list 
  I will start with a naive O(n^2) spatial complexity but store the average number of neighbors in a static variable
  At the end of the simulation I will print that value k. I want to make the next arry size 3/2 * k * n_particles and determine if that is acceptable
  I will add in a check for if an index is oob and call a resizing kernel to add k elements to the array and restart the neighbor search
*/

__host__ void allocateNeighborArrays(
  uint32_t **neighbors,
  uint32_t **neighbor_offset,
  uint32_t n_particles,
  uint32_t *list_size
) {
  (*list_size) = n_particles * (n_particles - 1);   // Naive O(n^2) memory calculation for now

  // Create memory
  cudaMallocManaged(neighbors, (*list_size) * sizeof(uint32_t));
  cudaMallocManaged(neighbor_offset, (n_particles + 1) * sizeof(uint32_t));
}
