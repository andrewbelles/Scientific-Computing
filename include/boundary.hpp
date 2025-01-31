#ifndef __BOUNDARY_HPP__
#define __BOUNDARY_HPP__

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "spatial.hpp"

/*
   Container structure that stores the bounds of an object that can interact with the simulation statically
   */
struct Container {
  float lower[3];
  float upper[3];
};

__host__ void updateBounds(spatialLookupTable *d_lookup_, sphParticle *d_particles_, std::vector<float> new_container, uint32_t particle_recount, const float h);

__host__ void callToBoundaryConditions(struct Container boundary, sphParticle *d_particles_, uint32_t n_particles, uint32_t n_partitions, const float h);


#endif // __BOUNDARY_HPP__
