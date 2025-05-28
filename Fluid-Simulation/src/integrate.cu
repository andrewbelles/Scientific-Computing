#include "integrate.hpp"

// Defines 
#define k 3000
#define rho0 1000
#define viscosity 1e-1

// #define __debug
// #define __verbose

// Offset table for kernel 
__constant__ int3 offset_table[27];

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
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__host__ __device__ static inline float3 operator-(const float3 a, const float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ static inline void operator-=(float3 &a, float3 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
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

__global__ static void neighbor_kernel(
  particleContainer *d_objs_,
  Lookup *d_lookup_,
  uint32_t n_partitions,
  uint32_t n_particles,
  const uint32_t containerBound[3],
  const float h
) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; 
    
  uint32_t pid = UINT32_MAX;
  uint32_t rel = 0, hash = 0, start = 0, end = 0;
  int3 cell_coord, relative_coord;

  __shared__ uint32_t shared_indices[256 * 48];
  const int MAX_NEIGHBORS = 48; 
  uint32_t *neighbors = &shared_indices[threadIdx.x * MAX_NEIGHBORS];
  int neighbor_count = 0;
  float3 local_pos, local_vel;

  if (idx < n_particles) {
    pid = d_lookup_->table_[idx].idx;
    assert(pid < n_particles);

    local_pos = make_float3(
      d_objs_->positions[pid],
      d_objs_->positions[pid + n_particles],
      d_objs_->positions[pid + 2 * n_particles]
    );

    cell_coord = positionToCellCoord(local_pos, h);

    for (int i = 0; i < 27 && neighbor_count < MAX_NEIGHBORS; i++) {
      relative_coord = cell_coord + offset_table[i];
      
      if (!inBounds(relative_coord, containerBound)) continue;

      hash  = hashPosition(relative_coord, n_partitions);
      assert(hash < n_partitions);
      start = d_lookup_->start_cell[hash];
      end   = d_lookup_->end_cell[hash];

      if (start == UINT32_MAX || end == UINT32_MAX) continue;
      assert(start < n_particles && end < n_particles + 1);

      for (uint32_t j = start; j < end && neighbor_count < MAX_NEIGHBORS; j++) {
        rel = d_lookup_->table_[j].idx;
        assert(rel < n_particles);

        if (rel == pid) continue;

        float3 relative_pos = make_float3(
          d_objs_->positions[rel],
          d_objs_->positions[rel + n_particles],
          d_objs_->positions[rel + 2 *n_particles]
        );

        float3 displace = local_pos - relative_pos;
        float distance = magnitude(displace);

        if (distance > 2.0 * h) continue;

        assert(neighbor_count < MAX_NEIGHBORS);
        neighbors[neighbor_count++] = rel;
      }
    }

    for (int i = 0; i < neighbor_count; i++) {
      rel = neighbors[i];
      assert(rel < n_particles);

      float3 relative_pos = make_float3(
        d_objs_->positions[rel],
        d_objs_->positions[rel + n_particles],
        d_objs_->positions[rel + 2 * n_particles]
      );

      float3 displace = local_pos - relative_pos;
      float distance = magnitude(displace);

      d_objs_->densities[pid] += d_objs_->masses[rel] * cubicSpline(distance, h);
    }

    // clamp density and calculate pressure
    d_objs_->densities[pid] = max(d_objs_->densities[pid], 0.1f * rho0);
    d_objs_->pressures[pid] = k * (d_objs_->densities[pid] - rho0);
  }  

  __syncthreads();

  if (idx < n_particles) {
    
    float3 pressure_force  = make_float3(0.0, 0.0, 0.0);
    float3 viscosity_force = make_float3(0.0, 0.0, 0.0); 

    local_vel = make_float3(
      d_objs_->velocities[pid],
      d_objs_->velocities[pid + n_particles],
      d_objs_->velocities[pid + 2 * n_particles]
    );

    for (int i = 0; i < neighbor_count; i++) {
      rel = neighbors[i];

      float3 relative_pos = make_float3(
        d_objs_->positions[rel],
        d_objs_->positions[rel + n_particles],
        d_objs_->positions[rel + 2 * n_particles]
      );

      float3 relative_vel = make_float3(
        d_objs_->velocities[rel],
        d_objs_->velocities[rel + n_particles],
        d_objs_->velocities[rel + 2 * n_particles]
      );

      float3 displace = local_pos - relative_pos;
      float distance = magnitude(displace);

      float3 direction = displace / (distance + tol);
      
      // Calculate the intermediate values for the pressure force
      float pressure_value = d_objs_->pressures[pid] / (d_objs_->densities[pid] * d_objs_->densities[pid]);
      pressure_value += d_objs_->pressures[rel] / (d_objs_->densities[rel] * d_objs_->densities[rel]);
      
      float common_term = d_objs_->masses[rel] * pressure_value * gradCubicSpline(distance, h);
      
      pressure_force -= (direction * common_term);
      viscosity_force += (((local_vel - relative_vel) * d_objs_->masses[rel] / d_objs_->densities[rel]) * laplacianCubicSpline(distance, h));
    }
    // Scale by viscosity constant 
    viscosity_force = viscosity_force * viscosity;

    d_objs_->pressure_forces[idx] = pressure_force.x;
    d_objs_->pressure_forces[idx + n_particles] = pressure_force.y;
    d_objs_->pressure_forces[idx + 2 * n_particles] = pressure_force.z;

    d_objs_->viscosity_forces[idx] = viscosity_force.x;
    d_objs_->viscosity_forces[idx + n_particles] = viscosity_force.y;
    d_objs_->viscosity_forces[idx + 2 * n_particles] = viscosity_force.z;
  }
}

/**
 * Host function to call the search kernel to find each particles forces relative to itself
 */
__host__ void neighborSearch(
  particleContainer *d_objs_,
  Lookup *d_lookup_,
  uint32_t n_partitions, 
  uint32_t n_particles,
  uint32_t containerCount[3],
  float h,
  bool first
) {
  //static uint32_t blocks = 0, threads = 0;
  //setGridSize(&blocks, &threads, n_particles);
  dim3 threads(256), blocks((n_particles + 256 - 1) / 256);
  cudaError_t err;

  neighbor_kernel<<<blocks, threads>>>(
    d_objs_,
    d_lookup_,
    n_partitions,
    n_particles,
    containerCount,
    h
  );
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Neighbor Failure: " << cudaGetErrorString(err) << '\n';
    return;
  }
}

/*
   Completes the first section of verlet integration
   */
__global__ void firstVerletKernel(particleContainer *d_objs_, uint32_t n_particles, float dt) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // Position and velocity update loop
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
    // Sums the pressure and viscosity forces for each axis
    // printf("idx %u repulsive force co %u : %f\n", idx, co, d_objs_->repulsive_forces[co]);
    forceSum[i] += (scale_factor * (d_objs_->pressure_forces[co] + d_objs_->viscosity_forces[co] + d_objs_->repulsive_forces[co]));
    
    // Integrates the velocity and position
    d_objs_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt)) * 0.99;
    d_objs_->positions[co] += (d_objs_->velocities[co] * static_cast<float>(dt));
  }
}

/*
   Second pass of verlet integration
   */
__global__ void secondVerletKernel(particleContainer *d_objs_, uint32_t n_particles, float dt) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // iterate over axis
  for (int i = 0; i < 3; ++i) {
    uint32_t co = idx + i * n_particles;
    // Sum forces from previous iteration 
    forceSum[i] += (scale_factor * (d_objs_->pressure_forces[co] + d_objs_->viscosity_forces[co]) / d_objs_->masses[idx] + d_objs_->repulsive_forces[co]);
    
    // Second half step to fully velocity
    d_objs_->velocities[co] += (forceSum[i] * static_cast<float>(0.5 * dt));
  }
}
