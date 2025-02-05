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

__host__ void updateBounds(Lookup *d_lookup_, particleContainer *d_objs_, std::vector<float> new_container, uint32_t particle_recount, const float h);

__host__ void callToBoundaryConditions(struct Container boundary, particleContainer *d_objs_, uint32_t n_particles, const float h);


#endif // __BOUNDARY_HPP__
