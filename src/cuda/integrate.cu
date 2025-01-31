#include "integrate.hpp"
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>

// Defines 
#define k 3000
#define rho0 1000
#define dt 1e-3
#define viscosity 1

// #define _debug
//#define _verbose

template <typename T>
#ifdef _debug 
__device__ static inline void relativeDisplacement(T a[3], const T b[3]) {
  for (int i = 0; i < 3; ++i) {
    printf("a[%d]: %f, b[%d]: %f\n", i, a[i], i, b[i]);
    a[i] -= b[i];
    printf("result: %f\n", a[i]);
  }
}
#else
__device__ static inline void relativeDisplacement(T a[3], const T b[3]) {
  for (int i = 0; i < 3; ++i)
    a[i] -= b[i];
}
#endif

template <typename T>
__device__ static inline float magnitude(T arr[3]) {
  return sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]);
}

template <typename T>
__device__ static inline void assign(T arr1[3], const T arr2[3]) {
  for (int i = 0; i < 3; ++i)
    arr1[i] = arr2[i];
}

template <typename T>
__device__ static inline void sum(T arr1[3], const T arr2[3]) {
  for (int i = 0; i < 3; ++i)
    arr1[i] += arr2[i];
}
/*
   Cubic Spline smooth field approximating kernel. 
   */
__host__ __device__ float cubicSpline(float distance, float smoothingRadius) {
  // Constant values
  const float  q = distance / smoothingRadius;
  const float  a3 = 1.0 / (M_PI * smoothingRadius * smoothingRadius * smoothingRadius);
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
__host__ __device__ float gradCubicSpline(float distance, float smoothingRadius) {
  const float  q = distance / smoothingRadius;
  const float  a3 = 1.0 / (M_PI * smoothingRadius * smoothingRadius * smoothingRadius);
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
__host__ __device__ float laplacianCubicSpline(float distance, float smoothingRadius) {
  const float  q = distance / smoothingRadius;
  const float  a3 = 1.0 / (M_PI * smoothingRadius * smoothingRadius * smoothingRadius);
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

/*
   Calculates the intermediate value from the displacement vector for the pressure force on a particle
   */
__device__ static void calculatePressureForce(float displacement[3], float rel_mass, float src_density, float rel_density, float kernel_val) { 
  for (int i = 0; i < 3; ++i) {
#ifdef _verbose
    printf("before disp[%d]: %f\n", i, displacement[i]);
#endif
    displacement[i] *= ((rel_mass * (src_density + rel_density)) / (2.0 * src_density * rel_density));
    displacement[i] *= -kernel_val;
#ifdef _verbose
    printf("pressure force [%d]: %f\n", i, displacement[i]);
#endif
  }
}

/*
   Calculates the intermediate value from the velocity difference vector for the viscosity force on a particle 
   */ 
__device__ static void calculateViscosityForce(float velocity_difference[3], float rel_mass, float kernel_val) {
  // intermediate_value = velocity_difference * (viscosity * particles[rel].mass * laplacianCubicSpline(distance, d_consts.h));
  for (int i = 0; i < 3; ++i) {
#ifdef _verbose
    printf("before vel_diff[%d]: %f\n", i, velocity_difference[i]);
#endif
    velocity_difference[i] *= (viscosity * rel_mass * kernel_val);
#ifdef _verbose 
    printf("visc force[%d]: %f\n", i, velocity_difference[i]);
#endif
  }
}

/*
   Kernel to determine the neighbors of each particle and the forces acting upon it
   */
__global__ static void neighborKernel(
  spatialLookupTable *d_lookup_,
  sphParticle *d_particles_,
  uint32_t n_partitions, 
  uint32_t n_particles,
  uint32_t containerCount[3],
  const float h 
) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return; 

  // Thread's particle id
  uint32_t pid = d_lookup_->table_[idx].idx;

  uint32_t cell_coord[3], relative_coord[3];
  uint32_t start, end, hash, rel;
  bool invalidCell = false; 

  float displacement[3], velocity_difference[3], distance = 0.0, kernel_val = 0.0;

  // print expected container maxes
#ifdef _debug
  printf("idx: %u\n", idx);
  if (idx == 0) {
    printf("cct: <%u, %u, %u>\n",
      containerCount[0],
      containerCount[1],
      containerCount[2]
    );
  }
#endif
  // Create cell coordinate at pid's position
  positionToCellCoord(cell_coord, d_particles_[pid].position, h);

  // Iterate around the cell coordinate
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        invalidCell = false; 
        int offset[3] = {dx, dy, dz};
#ifdef _debug
        printf("idx: %u, offset: <%d,%d,%d>\n", idx, offset[0], offset[1], offset[2]);
#endif
        // Find the relative cell coordinate for iter
        for (int i = 0; i < 3; ++i) {
          // Avoid unsigned int overflow
          if (cell_coord[i] == 0 && offset[i] == -1) {
            invalidCell = true;
            continue;
          }

          // Find relative cell coord 
          relative_coord[i] = cell_coord[i] + offset[i];
#ifdef _debug
          printf("idx: %u, rel: <%u,%u,%u>\n", 
            idx,
            relative_coord[0],
            relative_coord[1],
            relative_coord[2]
          );
#endif
          // Check if invalid
          if (relative_coord[i] >= containerCount[i]) {
            invalidCell = true;
          }
        }

        // If any of the relative coordinates are out of bounds
        if (invalidCell) continue;

        // Find relative cell coordinates hash value
        hash = hashPosition(relative_coord, n_partitions);
#ifdef _debug 
        printf("idx: %u, hash: %u\n", idx, hash);
#endif
        // Find start and stop indexes for lookup table
        start = d_lookup_->start_cell[hash];
        end   = d_lookup_->end_cell[hash];

        // Empty bucket 
        if (start > n_particles || end > n_particles) continue;
        if (start > end) {
          printf("Start/End discrepency\n");
          return;
        }
        // I want to assume that this works through this point (?) the third condition in the empty check above
        // theoretically shouldn't need to be there and it if it was true I'd want to exit(fail)
        // I guess I'm unsure that the hash function is correctly placing each particle into their respective bucket wo collision

        // Iterate over all particles in current bucket
        for (uint32_t i = start; i < end; ++i) {
          rel = d_lookup_->table_[i].idx; 
          if (rel >= n_particles) printf("wtf\n"); // shouldn't be an issue anymore -> so far hasn't been 
          if (rel == pid) continue; 

          // Copies position to displacement and finds the vector displacement between src and rel 
          assign(displacement, d_particles_[pid].position);
          relativeDisplacement(displacement, d_particles_[rel].position);
#ifdef _debug
          for (int i = 0; i < 3; ++i)
            if (displacement[i] > 10.0) printf("Error on idx %u for pid %u of displacement [%d]: %f\n", idx, pid, i, displacement[i]);
#endif
          distance = magnitude(displacement);       
#ifdef _debug 
          if (distance > 10.0) printf("Error on idx %u for pid %u!\n", idx, pid);
          printf("idx: %u, distance: %f\n", idx, distance);
#endif
          // Not neighbors -> continue 
          if (distance > (2.0 * h)) continue;  
#ifdef _debug
          printf("valid neighbor pair (pid, rel) -> (%u, %u)\n", pid, rel);
          printf("Neighbor Pair: (%u,%u)\n", pid, rel);
#endif
          // Find the vector difference of velocity
          assign(velocity_difference, d_particles_[pid].velocity);
          relativeDisplacement(velocity_difference, d_particles_[rel].velocity);

          // Find the density and set the new pressure
          d_particles_[pid].density += d_particles_[rel].mass * cubicSpline(distance, h);
          d_particles_[pid].pressure = static_cast<float>(k * (d_particles_[pid].density - rho0));

          // Calculate the approximate pressure force
          kernel_val = gradCubicSpline(distance, h);
          calculatePressureForce(displacement, d_particles_[rel].mass, d_particles_[pid].density, d_particles_[rel].density, kernel_val);
          sum(d_particles_[pid].pressure_force, displacement);

          // Calculate the approximate viscosity force 
          kernel_val = laplacianCubicSpline(distance, h);
          calculateViscosityForce(velocity_difference, d_particles_[rel].mass, kernel_val);
          sum(d_particles_[pid].viscosity_force, velocity_difference);
        }
      }
    }
  }
}

/*
   Host function to call the search kernel to find each particles forces relative to itself
   */
__host__ void callToNeighborSearch(
  spatialLookupTable *d_lookup_,
  sphParticle *d_particles_,
  uint32_t n_partitions, 
  uint32_t n_particles,
  uint32_t containerCount[3],
  const float h 
) {
  uint32_t threadsPerBlock = 256; 
  uint32_t gridSize = (n_partitions * threadsPerBlock - 1) / threadsPerBlock;

  // Call to kernel
  neighborKernel<<<gridSize, threadsPerBlock>>>(
    d_lookup_, d_particles_, n_partitions, n_particles, containerCount, h
  );
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(err) << '\n';
    exit(EXIT_FAILURE);
  }
}

/*
   Completes the first section of verlet integration
   */
__global__ void firstVerletKernel(sphParticle *d_particles_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  float forceSum[3] = {0.0, -9.81, 0.0};

  // Position and velocity update loop
  for (int i = 0; i < 3; ++i) {     
#ifdef _debug 
    printf("! before !\nidx: %u\n  pos: <%f,%f,%f>\n  vel: <%f,%f,%f>\n",
      idx,
      d_particles_[idx].position[0],
      d_particles_[idx].position[1],
      d_particles_[idx].position[2],
      d_particles_[idx].velocity[0],
      d_particles_[idx].velocity[1],
      d_particles_[idx].velocity[2]
    );
#endif
    // Sums the pressure and viscosity forces for each axis
    forceSum[i] += (d_particles_[idx].pressure_force[i] + d_particles_[idx].viscosity_force[i]);
    // Integrates the velocity and position
    d_particles_[idx].velocity[i] += (forceSum[i] * static_cast<float>(0.5 * dt));
    d_particles_[idx].position[i] += (d_particles_[idx].velocity[i] * static_cast<float>(dt));
#ifdef _debug 
    printf("! after !\nidx: %u\n  pos: <%f,%f,%f>\n  vel: <%f,%f,%f>\n",
      idx,
      d_particles_[idx].position[0],
      d_particles_[idx].position[1],
      d_particles_[idx].position[2],
      d_particles_[idx].velocity[0],
      d_particles_[idx].velocity[1],
      d_particles_[idx].velocity[2]
    );
#endif
  }
}

/*
   Second pass of verlet integration
   */
__global__ void secondVerletKernel(sphParticle *d_particles_, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
#ifdef _debug
  printf("Idx: %u\n", idx);
#endif
if (idx >= n_particles) return;

#ifdef _debug
  printf("Test Print\n");
  printf("Particle %u : %f\n", idx, d_particles_[idx].pressure_force[0]);
  printf("Seg fault?\n");
#endif 

  float forceSum[3] = {0.0, -9.81, 0.0};
  for (int i = 0; i < 3; ++i) {
    forceSum[i] += ((d_particles_[idx].pressure_force[i] + d_particles_[idx].viscosity_force[i]) / d_particles_[idx].mass);
    
#ifdef _debug
  printf("Force Sum %u: %f\n", idx, forceSum[i]); 
#endif 

    d_particles_[idx].velocity[i] += (forceSum[i] * static_cast<float>(0.5 * dt));
  }
}
