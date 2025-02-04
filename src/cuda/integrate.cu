#include "integrate.hpp"

// Defines 
#define k 3000
#define rho0 1000
#define dt 1e-3       // Change to dynamically shift in value 
#define viscosity 1e-2

// Offset table for kernel 
__constant__ int3 offset_table[27];
static float max_neighbors;

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

/**
 * Cubic Spline smooth field approximating kernel. 
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
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        // printf("%d %d %d\n", dz, dy, dx);
        host_offset[idx++] = make_int3(dx, dy, dz);  
      }
    }
  }

  // Copy the host table to the global constant offset table
  cudaMemcpyToSymbol(offset_table, host_offset, sizeof(host_offset));
}

__device__ static inline bool inBounds(const int3 a, const uint32_t bounds[3]) {
  if (a.x >= bounds[0] || a.x < 0 || a.y >= bounds[1] || a.y < 0 || a.z >= bounds[2] || a.z < 0)
    return false;
  else
    return true;
}

/**
 * Operator for int3 adding
 */
__device__ static inline int3 operator+(int3 a, int3 b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ neighborList *initNeighborList(uint32_t *list_size, uint32_t n_particles) {
  max_neighbors = (n_particles * 0.5);
  (*list_size) = static_cast<uint32_t>(n_particles * (max_neighbors - 1));

  neighborList *list = nullptr;

  cudaMallocManaged(&list->neighbors, (*list_size) * sizeof(int));
  cudaMallocManaged(&list->offsets, (n_particles + 1) * sizeof(int));
  cudaMallocManaged(&list->counts, n_particles * sizeof(int));

  cudaMallocManaged(&list, sizeof(neighborList));

  return list;
}

/*
 * Iterate over 3x3x3 centered by idx and determine number of neighbors for particle
 */
__global__ static void countNeighbors(
  neighborList *list,
  particleContainer *d_objs_,
  uint32_t list_size,
  Lookup *d_lookup_,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t containerCount[3],
  float h
) {

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 
  
  uint32_t pid = d_lookup_->table_[idx].idx;
  uint32_t rel = 0, hash = 0, start = 0, end = 0;
  int3 cell_coord, relative_coord;

  float3 relative_pos, displace;
  float3 local_pos = make_float3(
    d_objs_->positions[pid],
    d_objs_->positions[pid + n_particles],
    d_objs_->positions[pid + 2 * n_particles]
  );

  float distance = 0.0;

  // Find cell coordinate from pid position
  cell_coord = positionToCellCoord(local_pos, h);

  for (int i = 0; i < 27; ++i) {
    relative_coord = cell_coord + offset_table[i];

    // Check if in bounds 
    if (!inBounds(relative_coord, containerCount)) continue;

    // Hash relative coordinate and get start and end values 
    hash  = hashPosition(relative_coord, n_partitions);
    start = d_lookup_->start_cell[hash];
    end   = d_lookup_->end_cell[hash];

    // Empty bucket check
    if (start == UINT32_MAX || end == UINT32_MAX) continue;

    for (uint32_t j = start; j < end; ++j) {

      rel = d_lookup_->table_[j].idx;

      relative_pos = make_float3(
        d_objs_->positions[rel],
        d_objs_->positions[rel + n_particles],
        d_objs_->positions[rel + 2 * n_particles]
      );

      // Find distance 
      displace = local_pos - relative_pos;
      distance= magnitude(displace);

      if (distance > 2.0 * h) continue;

      list->counts[idx]++;
    }
  }
}

__global__ static void findNeighbors(
  neighborList *list,
  particleContainer *d_objs_,
  int status,
  uint32_t list_size,
  Lookup *d_lookup_,
  uint32_t n_partitions,
  uint32_t n_particles,
  uint32_t containerCount[3],
  float h
) {

  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 
  
  uint32_t pid = d_lookup_->table_[idx].idx;
  uint32_t count = 0, rel = 0, hash = 0, start = 0, end = 0;
  int3 cell_coord, relative_coord;

  float3 relative_pos, displace;
  float3 local_pos = make_float3(
    d_objs_->positions[pid],
    d_objs_->positions[pid + n_particles],
    d_objs_->positions[pid + 2 * n_particles]
  );

  float distance = 0.0;

  // Find cell coordinate from pid position
  cell_coord = positionToCellCoord(local_pos, h);

  for (int i = 0; i < 27; ++i) {
    relative_coord = cell_coord + offset_table[i];

    // Check if in bounds 
    if (!inBounds(relative_coord, containerCount)) continue;

    // Hash relative coordinate and get start and end values 
    hash  = hashPosition(relative_coord, n_partitions);
    start = d_lookup_->start_cell[hash];
    end   = d_lookup_->end_cell[hash];

    // Empty bucket check
    if (start == UINT32_MAX || end == UINT32_MAX) continue;

    for (uint32_t j = start; j < end; ++j) {

      rel = d_lookup_->table_[j].idx;

      relative_pos = make_float3(
        d_objs_->positions[rel],
        d_objs_->positions[rel + n_particles],
        d_objs_->positions[rel + 2 * n_particles]
      );

      // Find distance 
      displace = local_pos - relative_pos;
      distance= magnitude(displace);

      if (distance > 2.0 * h) continue;
      
      int ptr = list->offsets[idx] + count++;
      if (ptr >= list_size) {
        atomicExch(&status, 1);
        return;
      }

      list->neighbors[ptr] = rel; 
    }
  }
}

/**
 * Find the density, pressure, and system forces from the built neighbor list
 */
__global__ static void computeDensities(
  neighborList *list,
  particleContainer *d_objs_,
  uint32_t n_particles,
  float h
) { 
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 

  // In this function idx == pid therefore idxth value of neighbors 

  float3 local_pos = make_float3(
    d_objs_->positions[idx],
    d_objs_->positions[idx + n_particles],
    d_objs_->positions[idx + 2 * n_particles]
  );

  for (int i = list->offsets[idx]; i < list->offsets[idx + 1]; ++i) {

    uint32_t rel = list->neighbors[i];

    float3 relative_pos = make_float3(
      d_objs_->positions[rel],
      d_objs_->positions[rel + n_particles],
      d_objs_->positions[rel + 2 * n_particles] 
    );

    // Find distance
    float3 displace = local_pos - relative_pos;
    float distance  = magnitude(displace);

    // Find density sum
    d_objs_->densities[idx] += d_objs_->masses[rel] * cubicSpline(distance, h);
  }
  d_objs_->pressures[idx] = k * (d_objs_->densities[idx] - rho0);
}

__global__ static void computeForces(
  neighborList *list,
  particleContainer *d_objs_,
  uint32_t n_particles,
  const float h
) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 

  // Set local values
  float3 pressure_force  = make_float3(0.0, 0.0, 0.0);
  float3 viscosity_force = make_float3(0.0, 0.0, 0.0);

  float3 local_pos = make_float3(
    d_objs_->positions[idx],
    d_objs_->positions[idx + n_particles],
    d_objs_->positions[idx + 2 * n_particles]
  );
 
  float3 local_vel = make_float3(
    d_objs_->velocities[idx],
    d_objs_->velocities[idx + n_particles],
    d_objs_->velocities[idx + 2 * n_particles]
  );


  // Loop over neighbor indexes
  for (uint32_t i = list->offsets[idx]; i < list->offsets[idx + 1]; ++i) {
    // Set the relative particle id
    uint32_t rel = list->neighbors[i];
    
    float3 relative_pos = make_float3(
      d_objs_->positions[rel],
      d_objs_->positions[rel + n_particles],
      d_objs_->positions[rel + 2 * n_particles] 
    );

    // Find distance
    float3 displace = local_pos - relative_pos;
    float distance  = magnitude(displace);

    float3 relative_vel = make_float3( 
      d_objs_->velocities[rel],
      d_objs_->velocities[rel + n_particles],
      d_objs_->velocities[rel + 2 * n_particles]
    );

    // Unit vector of direction calculatio 
    float3 direction = displace / (distance + tol);

    // Calculate the intermediate values for the pressure force;
    float pressure_value  = d_objs_->pressures[idx] / (d_objs_->densities[idx] * d_objs_->densities[idx]);
    pressure_value       += d_objs_->pressures[rel] / (d_objs_->densities[rel] * d_objs_->densities[rel]); 
    float common_term    = d_objs_->masses[rel] * pressure_value * gradCubicSpline(distance, h);

    pressure_force  -= (direction * common_term);  
    viscosity_force += (((local_vel - relative_vel) * d_objs_->masses[rel] / d_objs_->densities[rel]) * laplacianCubicSpline(distance, h));
  }

  // Set forces 
  d_objs_->pressure_forces[idx] = pressure_force.x;
  d_objs_->pressure_forces[idx + n_particles] = pressure_force.y;
  d_objs_->pressure_forces[idx + 2 * n_particles] = pressure_force.z;
 
  d_objs_->viscosity_forces[idx] = viscosity_force.x;
  d_objs_->viscosity_forces[idx + n_particles] = viscosity_force.y;
  d_objs_->viscosity_forces[idx + 2 * n_particles] = viscosity_force.z;
}

__global__ static void resetCounts(int *counts, uint32_t n_particles) {
  for (int i = 0; i < n_particles; i++) {
    counts[i] = 0;
  }
}

/**
 * Host function to call the search kernel to find each particles forces relative to itself
 */
__host__ void neighborSearch(
  neighborList *list,
  particleContainer *d_objs_,
  Lookup *d_lookup_,
  uint32_t n_partitions, 
  uint32_t n_particles,
  uint32_t containerCount[3],
  uint32_t *list_size,
  float h
) {
  static uint32_t blocks = 0, threads = 0;
  setGridSize(&blocks, &threads, n_particles);
  int status = 0;
  cudaError_t err;

  // Perform computation until neighbor list is valid sized 
  do {
    countNeighbors<<<blocks, threads>>>(
      list,
      d_objs_,
      (*list_size), 
      d_lookup_,
      n_partitions,
      n_particles,
      containerCount,
      h
    );

    // Find prefix sum of list stored at offsets 
    thrust::exclusive_scan(thrust::device, list->counts, list->counts + n_particles, list->offsets);

    findNeighbors<<<blocks, threads>>>(
      list,
      d_objs_,
      status,
      (*list_size), 
      d_lookup_,
      n_partitions,
      n_particles,
      containerCount,
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
      max_neighbors = max_neighbors / n_particles + max_neighbors * 0.5;
      (*list_size) = static_cast<uint32_t>(n_particles * (max_neighbors - 1));
      cudaFree(list->neighbors);
      cudaMallocManaged(&list->neighbors, (*list_size) * n_particles * sizeof(uint32_t));
    }

  } while (status != 0); 

  computeDensities<<<blocks, threads>>>(
    list,
    d_objs_,
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

  // Expected success 
  computeForces<<<blocks, threads>>>(
    list,
    d_objs_,
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

  resetCounts<<<1, 1>>>(list->counts, n_particles);
}

/*
   Completes the first section of verlet integration
   */
__global__ void firstVerletKernel(particleContainer *d_objs_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // Position and velocity update loop
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
   
    // Sums the pressure and viscosity forces for each axis
    forceSum[i] += (d_objs_->pressure_forces[co] + d_objs_->viscosity_forces[co]);
    
    // Integrates the velocity and position
    d_objs_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt));
    d_objs_->positions[co] += (d_objs_->velocities[co] * static_cast<float>(dt));
  }
}

/*
   Second pass of verlet integration
   */
__global__ void secondVerletKernel(particleContainer *d_objs_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // iterate over axis
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
    // Sum forces from previous iteration 
    forceSum[i] += ((d_objs_->pressure_forces[co] + d_objs_->viscosity_forces[co]) / d_objs_->masses[idx]);
    
    // Second half step to fully velocity
    d_objs_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt));
  }
}
