#ifndef __ITERATE_HPP__
#define __ITERATE_HPP__

#include "spatial.hpp"
#include "boundary.hpp"
#include "integrate.hpp"

__host__ void particleIterator(
  spatialLookupTable *d_lookup_,
  sphParticle *d_particles_,
  float **u_positions,
  float **u_densities,
  std::vector<float> container,
  uint32_t n_particles,
  uint32_t n_partitions, 
  const float h
);

#endif // __ITERATE_HPP__
