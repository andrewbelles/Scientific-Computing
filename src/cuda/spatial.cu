#include "spatial.hpp"
#include <cstdlib>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <surface_types.h>

//#define _debug
//#define _verbose
#define _errorcheck

/**
 * Simple templated function to return the minimum of any Vector
 */
template <typename T> 
T min(const std::vector<T>& vec) {
   
  if (vec.empty()) {
    std::cerr << "Empty Vector -> No val" << "\n";
  }

  // Checks if proceeding values are less than first 
  T res = vec[0];
  for (auto& val : vec) {
    res = (val < res) ? val : res;
  }
  return res;
}

/**
 * Simple templated function to return the maximum of any Vector
 */
template <typename T>
T max(const std::vector<T>& vec) {
  if (vec.empty()) {
    std::cerr << "Empty Vector -> No val" << "\n";
  }

  // Checks if proceeding values are greater than first 
  T res = vec[0];
  for (auto& val : vec) {
    res = (val > res) ? val : res;  
  }
  return res;
}

/**
 * Generic bitshift function to return the next power of two 
 * Restricted to unsigned integer types
 */
template <typename T>
T findSquare(T value) {
  // Base case (If particle_count was zero for some reason)
  if (value == 0) {
    return 1;
  }

  // Maniputate bits to be 2^(n) - 1
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;

  // Re-add 1 and return result
  return value + 1;
}

/**
 * Creates the cell coordinate from the relative position of a particle's position 
 */
__device__ void positionToCellCoord(uint32_t cell_coord[3], const float position[3], const float h) {
  for (int i = 0; i < 3; ++i) {
    cell_coord[i] = static_cast<uint32_t>(floor(position[i] / h));
  }
}

/**
 * Hashes a particles cell coordinate to a hash index
 */
__device__ uint32_t hashPosition(const uint32_t cell_coord[3], uint32_t n_partitions) {
  const uint32_t primes[] = {73856093, 19349663, 83492791};
  uint32_t hash = 0;

  // Distribute from prime values
  for (int i = 0; i < 3; ++i) {
    hash += cell_coord[i] * primes[i];
  }
  // Return hash fit into partition size
  return hash % n_partitions;
}


__global__ static void fillSentinelKernel(spatialLookupTable *d_lookup_, uint32_t n_partitions) { 
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_partitions) return;

  d_lookup_->start_cell[idx] = UINT32_MAX;
  d_lookup_->end_cell[idx]   = UINT32_MAX;
}

/**
 * Kernel to quickly insert all particle positions into the lookup hashmap unordered
 */
__global__ static void insertTableKernel(
    spatialLookupTable *d_lookup_,
    particleContainer *d_particleContainer_,
    uint32_t n_particles,
    uint32_t n_partitions,
    uint32_t padded_size,
    const float h
) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;
  // Find hash value 
  uint32_t cell_coord[3];
  
  float local_pos[3] = {
    d_particleContainer_->positions[idx],
    d_particleContainer_->positions[idx + n_particles],
    d_particleContainer_->positions[idx + 2 * n_particles]
  };

  positionToCellCoord(cell_coord, local_pos, h);
  uint32_t hash = hashPosition(cell_coord, n_partitions);

  // Fill table for idx
  if (idx >= padded_size) {  
    d_lookup_->start_cell[idx] = UINT32_MAX;
    d_lookup_->end_cell[idx]   = UINT32_MAX;
  } else {
    d_lookup_->table_[idx].idx      = idx;
    d_lookup_->table_[idx].cell_key = hash;
  }
}

/**
 * Kernel Function to sort lookup table by cell_key value. 
 * Credit to github user rgba for the relative structure of pair selectioon
 */
__global__ static void sortPairs(spatialLookupTable *d_lookup_, int j, int i, uint32_t paddedSize) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t idxj = idx ^ j;
  if (idx >= paddedSize) return;
  struct tableEntry *d_table_ = d_lookup_->table_;
  struct tableEntry temp;
  
  if (idxj > idx) {
    if ((idx & i) == 0) {
      if (d_table_[idx].cell_key > d_table_[idxj].cell_key) {
        temp = d_table_[idx];
        d_table_[idx] = d_table_[idxj];
        d_table_[idxj] = temp;
      }
    } else {
      if (d_table_[idx].cell_key < d_table_[idxj].cell_key) {
        temp = d_table_[idx];
        d_table_[idx] = d_table_[idxj];
        d_table_[idxj] = temp;
      }
    }
  }
}

/**
 * Sorts array of struct tableEntry by their cell cell_key. 
 * Optimized for parallelization on GPU
 */
__host__ void bitonicSort(spatialLookupTable *d_lookup_, uint32_t paddedSize) {
  // Determine the number of threads 
  int threadPerBlock = 256;
#ifdef _debug
  std::cout << "paddedSize: " << paddedSize << '\n';
#endif
  int blocks = (paddedSize + threadPerBlock - 1) / threadPerBlock;
#ifdef _debug
  std::cout << "pre-kernel index loop\n";
#endif
  // Iterator to ensure pairs are correctly sized 
  for (uint32_t i = 2; i <= paddedSize; i <<= 1) {
    for (uint32_t j = i >> 1; j > 0; j >>= 1) {
      // Call GPU kernel
      sortPairs<<<blocks, threadPerBlock>>>(d_lookup_, j, i, paddedSize);
    }
  }
}

/**
 * Create the start and end cell based on filled lookup tableEntry
 */
__global__ static void setTableIndexes(spatialLookupTable *d_lookup_, uint32_t n_partitions, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + 1; // Idx: omit 0
  if (idx >= n_particles) return;
#ifdef _debug
  printf("idx %u\n", idx);
#endif
  // Ensure only first thread completes the task of initializing first values
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_lookup_->end_cell[d_lookup_->table_[n_particles - 1].cell_key] = n_particles;
    d_lookup_->start_cell[d_lookup_->table_[0].cell_key]             = 0; 
  }
 
  // Pull keys from current idx 
  uint32_t curr_key = d_lookup_->table_[idx].cell_key;
  uint32_t prev_key = d_lookup_->table_[idx - 1].cell_key;

  // Set the next value if on "exchange" point
  if (curr_key != prev_key) {
    d_lookup_->end_cell[prev_key]   = idx;
    d_lookup_->start_cell[curr_key] = idx; 
  }
}

__global__ void printTable(spatialLookupTable *d_lookup_, uint32_t n_partitions, uint32_t paddedSize) {
  printf(">> Sorted Spatial Lookup Table\n");
  for (uint32_t idx = 0; idx < paddedSize; ++idx) {
    printf(" Table ID: %u\n", idx);
    printf(" Cell Key: %u\n", d_lookup_->table_[idx].cell_key);
    printf(" Particle: %u\n", d_lookup_->table_[idx].idx);
  }
  printf(">> Start and End Cell Tables\n");
  for (uint32_t idx = 0; idx < n_partitions; ++idx) {
    printf(" Start for Hash %u: %u\n", idx, d_lookup_->start_cell[idx]);
    printf(" End for Hash %u: %u\n", idx, d_lookup_->end_cell[idx]);  
  }
}

/**
 * Host function to call individual kernels to set values of table 
 */
__host__ void hostFillTable(spatialLookupTable *d_lookup_, particleContainer *d_particleContainer_, uint32_t n_partitions, uint32_t n_particles, uint32_t paddedSize, const float h) {
  // Might not be optimal as there are a maximum number of threads and is inflexible
  uint32_t threadsPerBlock = 256;
  uint32_t insertSize = (n_partitions + threadsPerBlock - 1) / threadsPerBlock;
  uint32_t tableSize = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err;

  fillSentinelKernel<<<insertSize, threadsPerBlock>>>(d_lookup_, n_partitions);

  // Call to kernel to fill table
  insertTableKernel<<<tableSize, threadsPerBlock>>>(
    d_lookup_,
    d_particleContainer_,
    n_particles,
    n_partitions,  
    paddedSize,
    h
  );

#ifdef _debug
  std::cout << "call to insert kernel complete\n";
#endif
  // Sort table bitonically
  bitonicSort(d_lookup_, paddedSize);
#ifdef _debug
  std::cout << "sorted lookup\n";
#endif
  // Set the start and end cell arrays
  setTableIndexes<<<tableSize, threadsPerBlock>>>(d_lookup_, n_partitions, n_particles);

  cudaDeviceSynchronize();
#ifdef _errorcheck
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Set Error: " << cudaGetErrorString(err) << '\n';
  }
#endif

#ifdef _verbose
  printTable<<<1, 1>>>(d_lookup_, n_partitions, paddedSize);
#endif
  
}

/**
 * Essentially a constructor for the simulation
 */
__host__ void initalizeSimulation(
    spatialLookupTable **d_lookup_,
    particleContainer **d_particleContainer_,
    const std::vector<float> container,
    uint32_t *n_partitions,
    uint32_t n_particles,
    const float h
) {
  // Ensure GPU is recognized...

  cudaError_t err;
  int deviceCount, device_id;
  err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    std::cerr << "Device Error: " << cudaGetErrorString(err) << '\n';
  }
#ifdef _debug
  else {
    std::cout << "Number of CUDA devices: " << deviceCount << '\n';
  }
#endif 
  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
#ifdef _debug
    std::cout << "Device " << i << ": " << prop.name << '\n';
#endif
    device_id = i;
  }

  // Set used device
  cudaSetDevice(device_id);
  
  // Maximum and Minimum values that the simulation can be initialized to. 
  /*float maximum = min(container) - 1.0;
  float minimum = maximum - (max(container) - min(container));*/ 
  // Hard coded values
  float maximum = 9.0;
  float minimum = 1.0;
#ifdef _debug
  std::cout << "max: " << maximum << " min: " << minimum << '\n';
#endif
  // Generate the axis count for each 
  std::vector<uint16_t> containerID(3, 0);
  (*n_partitions) = 1;
  for (int i = 0; i < 3; ++i) {
    containerID[i] = static_cast<uint16_t>(floor(container[i] / h));
    (*n_partitions) *= static_cast<uint32_t>(containerID[i]);
  }
#ifdef _debug
  std::cout << "found container maximums\n";
#endif
  // Get size of arr on order 2^n
  uint32_t paddedSize = findSquare(n_particles); 
  
  // Initialize lookup table in device memory
  spatialLookupTable lookup_, host_lookup_;

  // Malloc Memory on the device for lookup members
  cudaMalloc(&lookup_.table_, paddedSize * sizeof(tableEntry));
  cudaMalloc(&lookup_.start_cell, (*n_partitions) * sizeof(uint32_t));
  cudaMalloc(&lookup_.end_cell, (*n_partitions) * sizeof(uint32_t));
  // Malloc Memory for ptr to device lookup
  cudaMalloc(d_lookup_, sizeof(spatialLookupTable));

  // Set device ptrs in host
  host_lookup_.table_     = lookup_.table_;
  host_lookup_.start_cell = lookup_.start_cell;
  host_lookup_.end_cell   = lookup_.end_cell;
  
  // Copy memory to device from host
  cudaMemcpy(
    (*d_lookup_),
    &host_lookup_,
    sizeof(spatialLookupTable),
    cudaMemcpyHostToDevice
  );

  // sphParticle initalize
#ifdef _debug
  std::cout << "Pre particle malloc\n";  
#endif
  /*
     Ptr is allocated as a managed ptr which will be auto migrated 
     to the GPU once it is dereferenced on the GPU. It will stay there unless
     dereferenced
  */

  // Coalesced device ptrs 
  float *u_pos, *u_vel, *u_prf, *u_visf, *u_mass, *u_dens, *u_pr;
  cudaMallocManaged(&u_pos, n_particles * 3 * sizeof(float));
  cudaMallocManaged(&u_vel, n_particles * 3 * sizeof(float));
  cudaMallocManaged(&u_prf, n_particles * 3 * sizeof(float));
  cudaMallocManaged(&u_visf, n_particles * 3 * sizeof(float));
  cudaMallocManaged(&u_mass, n_particles * sizeof(float));
  cudaMallocManaged(&u_dens, n_particles * sizeof(float));
  cudaMallocManaged(&u_pr, n_particles * sizeof(float));

  // Refactor: handle particle construction differently for SoA initialization
  cudaMallocManaged(d_particleContainer_, sizeof(particleContainer));

  // Set only value in d_particleContainer_ to constructor
  new (*d_particleContainer_) particleContainer(
    u_pos, 
    u_vel,
    u_prf,
    u_visf,
    u_mass,
    u_dens,
    u_pr,
    n_particles,
    minimum,
    maximum
  );
  

#ifdef _debug
  std::cout << "Constructed array\n";
#endif
#ifdef _debug
  std::cout << "Test Mass: " << (*d_particleContainer_)->masses[0] << '\n'; // Check if constructed
#endif

  // If this seg faults the program its on the gpu (maybe?)
#ifdef _debug
  std::cout << "Test Mass: " << (*d_particleContainer_)->masses[0] << '\n'; // Check if constructed
  std::cout << "Check complete\n";
#endif

  // Call to host function to fill table with values 
  hostFillTable(
    (*d_lookup_),
    (*d_particleContainer_),
    (*n_partitions),
    n_particles,
    paddedSize,
    h
  );
}
