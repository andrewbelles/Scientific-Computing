#include "boundary.hpp"

__host__ static inline bool equal(std::vector<float> new_container, std::vector<float> boundary) {
  for (int i = 0; i < 3; ++i) {
    if (new_container[i] != boundary[i]) return false;
  }
  return true; 
} 

/*
   Updates the bounds given a new container vector. This stalls the program until
   it can accurately update the spatial hashmap for a new size. 
   */
__host__ void updateBounds(Lookup *d_lookup_, particleContainer *d_objs_, std::vector<float> new_container, uint32_t particle_recount, const float h) {
  // refac cuda mem manage now pass cpu resize 
  static std::vector<float> boundary; 
  static uint32_t n_particles  = 0;
  static uint32_t n_partitions = 0;

  bool setLookup = false;

  // Set static size variables

  // Set boundary size
  if (boundary.size() == 0) {
    boundary = std::vector<float>(3, 0);
    boundary = new_container;
  }

  // Set partition count
  if (n_partitions == 0) {
    // Count the number of partitions ("volume")
    uint32_t partition_counter[3];
    n_partitions = 1;
    for (int i = 0; i < 3; ++i) {
      partition_counter[i] = static_cast<uint32_t>(float(boundary[i] / (2.0 * h)));
      n_partitions *= partition_counter[i];
    }
    n_partitions = convertToPrime(n_partitions);
  }

  // Set particle count
  if (!n_particles) {
    n_particles = particle_recount;
  } 

  // Resize particle related structs 
  if (particle_recount != n_particles) {

    // Change lookup table size
    if (findSquare(particle_recount) != findSquare(n_particles))
      setLookup = true;
    
    float minimum = 2 * h;
    float maximum = boundary[0] - 2 * h;  // Hardcoded for now

    // We can pull the position and velocity vectors and then delete everything else and create a new ptr
    float *n_pos, *n_vel;
    
    // Create new memory 
    cudaMallocManaged(&n_pos, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_vel, particle_recount * 3 * sizeof(float));
    
    // Set copy size
    uint32_t copy = (n_particles < particle_recount) ? n_particles : particle_recount;

    cudaMemcpy(n_pos, d_objs_->positions, copy * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(n_vel, d_objs_->velocities, copy * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    delete (d_objs_);

    float *n_prf, *n_visf, *n_repf, *n_mass, *n_dens, *n_pr;

    new (d_objs_) particleContainer();

    // Allocate accumulators
    cudaMallocManaged(&n_prf, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_visf, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_repf, particle_recount * 3 * sizeof(float));
    cudaMallocManaged(&n_mass, particle_recount * sizeof(float));
    cudaMallocManaged(&n_dens, particle_recount * sizeof(float));
    cudaMallocManaged(&n_pr, particle_recount * sizeof(float));

    // Add new particles if size is larger
    if (particle_recount > n_particles) {
      d_objs_->addNewParticles(
        n_pos,
        n_vel,
        n_particles, 
        particle_recount,
        minimum,
        maximum
      );
    }

    d_objs_->slowSetAccumulators(
      n_prf,
      n_visf,
      n_repf,
      n_mass,
      n_dens,
      n_pr,
      particle_recount
    );
  }

  // Resize container related structs
  if (!equal(new_container, boundary) || setLookup) {
    // Not worrying about it right now
  }
}

/*
 * Kernel call to naively, individually rectify potential out of bounds behavior for a particle 
 */
__global__ static void boundaryKernel(const struct Container boundary, particleContainer *d_objs_, uint32_t n_particles, const float smooth_radius) {
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_particles) return;
  const float strength = 1000, cutoff = 2 * smooth_radius, decay_length = 0.1 * cutoff;

  // Calculate the repulsive force in each direction from boundary
  float3 repulsive_potential, position = make_float3(
    d_objs_->positions[idx],
    d_objs_->positions[idx + n_particles],
    d_objs_->positions[idx + 2 * n_particles]
  );
  float3 upper_distance = make_float3(
    boundary.upper[0],
    boundary.upper[1],
    boundary.upper[2]
  );
  
  // Array of pointers to float3 struct 
  float *rp[]  = {&repulsive_potential.x, &repulsive_potential.y, &repulsive_potential.z};
  float *pos[] = {&position.x, &position.y, &position.z};
  float *upp[] = {&upper_distance.x, &upper_distance.y, &upper_distance.z};
  
  // Loop over each axis 
  for (int i = 0; i < 3; ++i) {
    *upp[i] -= *pos[i];

    // Store position: If the distance to upper bound is less than the true position than it is closer to the upper bound than the lower
    // *The lower bound is defined at 0.0 for each axii
    float position = (*upp[i] < *pos[i]) ? (*upp[i]) : *pos[i];
    float A = (*upp[i] < *pos[i]) ? -1.0 * strength : strength;

    // Find the potential and divide by decay length for force 
    *rp[i] = (position < cutoff) ? A * exp(-(position - cutoff) / decay_length) : 0.0;
    d_objs_->repulsive_forces[idx + i * n_particles] = *rp[i] / decay_length;
  } 
}

/*
 * Host call to handle the call to boundaryKernel 
 * Sets the range of acceptable thread idx to act on particles 
 */
__host__ void callToBoundaryConditions(struct Container boundary, particleContainer *d_objs_, uint32_t n_particles, const float h) {

  static uint32_t blocks = 0, threads = 0;
  setGridSize(&blocks, &threads, n_particles);

  // Iterate over each container and call kernel for each one 
  boundaryKernel<<<blocks, threads>>>(
    boundary,
    d_objs_,
    n_particles,
    h
  );
  cudaDeviceSynchronize();
}
