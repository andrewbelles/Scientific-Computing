#ifndef __ITERATE_HPP__
#define __ITERATE_HPP__

#include "spatial.hpp"
#include "boundary.hpp"
#include "integrate.hpp"

#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

__host__ void particleIterator(
  particleContainer *d_objs_,
  float*& u_positions,
  float*& u_densities,
  Lookup *d_lookup_,
  const std::vector<float> container,
  const uint32_t n_particles,
  const uint32_t n_partitions, 
  const float h,
  bool first 
);

#endif // __ITERATE_HPP__
