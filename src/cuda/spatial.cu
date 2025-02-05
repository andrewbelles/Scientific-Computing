#include "spatial.hpp"

/**
 * Refactor: 
 * Adjust code s.t each source file has globally defined block sizes that are calculated once and stay static 
 * Maybe this isn't a great idea? We can just declare static at the top of a kernel call as the redundant calculation is not impactful
 */

/**
 * Checks if static grid size vars are zero and the appropriately sets them to maximize SM workload
 * Inlined; Should this be static? 
 */
__host__ inline void setGridSize(uint32_t *blocks, uint32_t *threads, uint32_t arr_size) {
  const uint16_t expected_threads = 256;

  (*blocks) = ((*blocks) == 0) 
    ? ((*blocks) = ((arr_size + expected_threads - 1) / expected_threads) < 30) 
       ? 30 
       : ((arr_size + expected_threads - 1) / expected_threads) 
    : (*blocks);
  (*threads) = ((*threads) == 0) ? findSquare((arr_size - 1) / ((*blocks) - 1)) : (*threads);
#ifdef __debug
  std::cout << "Set grid sizes to: (" << (*blocks) << " x " << (*threads) << ")\n";
#endif
}

/*
 * Simple boolean function to return if prime or not. From chat gpt-4o
 */
template <typename T>
bool isPrime(T val) {
  if (val <= 1) {
        return false;
    }
    if (val <= 3) {
        return true;
    }
    if (val % 2 == 0 || val % 3 == 0) {
        return false;
    } 
    for (int i = 5; i * i <= val; i += 6) {
        if (val % i == 0 || val % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

/*
 * Increments value till its prime : Inefficient I know
 */
template <typename T>
void convertToPrime(T *val) {
  while (!isPrime(*val)) {
    (*val)++;
  }
}

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
__device__ int3 positionToCellCoord(float3 position, const float h) {
  int3 cell_coord;

  cell_coord.x = floorf(position.x / (2.0 * h));
  cell_coord.y = floorf(position.y / (2.0 * h));
  cell_coord.z = floorf(position.z / (2.0 * h));

  return cell_coord;
}

/**
 * Hashes a particles cell coordinate to a hash index
 */
__device__ uint32_t hashPosition(int3 cell_coord, uint32_t n_partitions) {
  const uint64_t primes[] = {73856093, 19349663, 83492791};   // Local hash primes array
  uint32_t hash = 0;

  // Distribute from prime values
  // Promote int32_t to 64_t by multiplying by uint64_t
  hash = cell_coord.x * primes[0];
  hash += cell_coord.y * primes[1];
  hash += cell_coord.z * primes[2]
  ;
  // Return hash fit into partition size
  return hash % n_partitions;
}

/**
 * Kernel to quickly fill all values in start and end cell arrays to sentinel values
 */
__global__ static void fillSentinelKernel(Lookup *d_lookup_, uint32_t n_partitions) { 
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_partitions) return;

  d_lookup_->start_cell[idx] = UINT32_MAX;
  d_lookup_->end_cell[idx]   = UINT32_MAX;
}

/**
 * Kernel to quickly insert all particle positions into the lookup hashmap unordered
 */
__global__ static void insertTableKernel(
    Lookup *d_lookup_,
    particleContainer *d_particleContainer_,
    uint32_t n_particles,
    uint32_t n_partitions,
    uint32_t padded_size,
    const float h
) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;

  // Find hash value  
  float3 local_pos = make_float3(
    d_particleContainer_->positions[idx],
    d_particleContainer_->positions[idx + n_particles],
    d_particleContainer_->positions[idx + 2 * n_particles]
  );

  int3 cell_coord = positionToCellCoord(local_pos, h);
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
__global__ static void sortPairs(Lookup *d_lookup_, int j, int i, uint32_t paddedSize) {
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
__host__ void bitonicSort(Lookup *d_lookup_, uint32_t padded_size) {
  // Determine the number of threads dynamically to ensure SM workload maximized
  static uint32_t threads = 0;
  static uint32_t blocks = 0;

  setGridSize(&blocks, &threads, padded_size); // Dynamic check of grid size 

  // Iterator to ensure pairs are correctly sized 
  for (uint32_t i = 2; i <= padded_size; i <<= 1) {
    for (uint32_t j = i >> 1; j > 0; j >>= 1) {
      // Call GPU kernel
      sortPairs<<<blocks, threads>>>(d_lookup_, j, i, padded_size);
    }
  }
}

/**
 * Create the start and end cell based on filled lookup tableEntry
 */
__global__ static void setTableIndexes(Lookup *d_lookup_, uint32_t n_partitions, uint32_t n_particles) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x + 1; // Idx: omit 0
  if (idx >= n_particles) return;
  
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

__global__ static void printTable(Lookup *d_lookup_, uint32_t n_partitions, uint32_t n_particles) {
  for (uint32_t i = 0; i < n_particles; ++i) {
    printf("TABLE %u\n", i);
    printf("  IDX:  %u\n", d_lookup_->table_[i].idx);
    printf("  CELL: %u\n", d_lookup_->table_[i].cell_key);
  }

  for (uint32_t i = 0; i < n_partitions; ++i) {
    printf("CELL %u\n", i);
    printf("  START: %u\n", d_lookup_->start_cell[i]);
    printf("  END  : %u\n", d_lookup_->end_cell[i]);
  }
}

/**
 * Host function to call individual kernels to set values of table 
 */
__host__ void hostFillTable(Lookup *d_lookup_, particleContainer *d_particleContainer_, uint32_t n_partitions, uint32_t n_particles, uint32_t padded_size, const float h) {
  // Might not be optimal as there are a maximum number of threads and is inflexible
  static uint32_t insert_threads = 0, table_threads = 0;
  static uint32_t insert_blocks = 0, table_blocks = 0;

  static int iter;
  if (insert_blocks == 0) iter = 0;

  setGridSize(&insert_blocks, &insert_threads, n_partitions);
  setGridSize(&table_blocks, &table_threads, padded_size);

  fillSentinelKernel<<<insert_blocks, insert_threads>>>(d_lookup_, n_partitions);

  // Call to kernel to fill table
  insertTableKernel<<<table_blocks, table_threads>>>(
    d_lookup_,
    d_particleContainer_,
    n_particles,
    n_partitions,  
    padded_size,
    h
  );

  // Sort table bitonically
  bitonicSort(d_lookup_, padded_size);
  
  // Set the start and end cell arrays
  setTableIndexes<<<table_blocks, table_threads>>>(d_lookup_, n_partitions, n_particles);

  if (iter == 0)
    printTable<<<1 ,1>>>(d_lookup_, n_partitions, n_particles);

  cudaDeviceSynchronize();
  iter++;
}

/**
 * Essentially a constructor for the simulation
 */
__host__ void initalizeSimulation(
  Lookup **d_lookup_,
  particleContainer **d_particleContainer_,
  const std::vector<float> container,
  uint32_t *n_partitions,
  uint32_t n_particles,
  const float h
) {
  // Maximum and Minimum values that the simulation can be initialized to. 
  /*float maximum = min(container) - 1.0;
  float minimum = maximum - (max(container) - min(container));*/ 
  // Hard coded values
  float maximum = 2 * h;
  float minimum = container[0] - 2 * h;

  // Generate the axis count for each 
  std::vector<uint16_t> containerID(3, 0);
  (*n_partitions) = 1;
  for (int i = 0; i < 3; ++i) {
    containerID[i] = static_cast<uint16_t>(floor(container[i] / (2.0 * h)));
    (*n_partitions) *= static_cast<uint32_t>(containerID[i]);
  }

  convertToPrime(n_partitions);
 
  // Get size of arr on order 2^n
  uint32_t paddedSize = findSquare(n_particles); 
  
  // Initialize lookup table in device memory
  Lookup lookup_, host_lookup_;

  // Malloc Memory on the device for lookup members
  cudaMalloc(&lookup_.table_, paddedSize * sizeof(tableEntry));
  cudaMalloc(&lookup_.start_cell, (*n_partitions) * sizeof(uint32_t));
  cudaMalloc(&lookup_.end_cell, (*n_partitions) * sizeof(uint32_t));
  // Malloc Memory for ptr to device lookup
  cudaMalloc(d_lookup_, sizeof(Lookup));

  // Set device ptrs in host
  host_lookup_.table_     = lookup_.table_;
  host_lookup_.start_cell = lookup_.start_cell;
  host_lookup_.end_cell   = lookup_.end_cell;
  
  // Copy memory to device from host
  cudaMemcpy(
    (*d_lookup_),
    &host_lookup_,
    sizeof(Lookup),
    cudaMemcpyHostToDevice
  );

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
