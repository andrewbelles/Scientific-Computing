#include <cmath>
#include <cuda_runtime.h>
#include <cublas.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

using std::min;

struct ParticleMatrix {
  // metadata for gpu 
  size_t cols;
  size_t rows;    // Equivalent to the # of stored state information
  float mass; 
  // Plane boundaries should be managed on gpu (?)

  // elements of matrix 
  float* rho;
  float3* x;
  float3* v;
  float3* fp;
  float3* fvi;
  float3* fsys;
  
  ParticleMatrix(const size_t& columns, const size_t& rows) : cols(columns), rows(rows) 
  {
    mass = 1.0 / columns;   // Expected total mass / particle count  

    // Create memory for each particle
    cudaMalloc(&rho, cols * sizeof(float));
    cudaMalloc(&x, cols * sizeof(float3));
    cudaMalloc(&v, cols * sizeof(float3));
    cudaMalloc(&fp, cols * sizeof(float3));
    cudaMalloc(&fvi, cols * sizeof(float3));
    cudaMalloc(&fsys, cols * sizeof(float3));
  }

  // Free gpu memory 
  ~ParticleMatrix()
  {
    cudaFree(rho);
    cudaFree(x);
    cudaFree(v);
    cudaFree(fp);
    cudaFree(fvi);
    cudaFree(fsys);
  }

}; // Each array initialized to same size on gpu

// Fuck fuck fuck fuck 
// void* A_ptr = nullptr; 
// cudaMallocManaged(&A_ptr, sizeof(ParticleMatrix));

// Symbolically linked Matrix struct that contains the metadata of a gpu managed structure
enum class Operation : uint8_t { None, Transpose, Conjugate };
enum class Layout : uint8_t { Row, Col };

struct MatrixData
{
  size_t cols;
  size_t rows;
  size_t lda; 
  // cudaStream_t stream{0};   // Maybe? 
  
  // Enums 
  Layout layout;
  Operation op;

  MatrixData(const size_t& cols, const size_t& rows/*, const cudaStream_t& stream = 0*/) 
    : cols(cols), rows(rows)/*, stream(stream)*/ {}
};

// Define the domain under which particles will be initialized etc 
// Redefine this to operate as a struct so that it can be passed onto gpu
struct Domain 
{
  float3 extrema[2];
  float spacing{0.0};
  float min; 
  float h; 

  __host__ __device__
  Domain() {}

  __host__ 
  Domain(const float3& max, const size_t& particles_along_axis = 50)
  {
    // Extrema values of plane (from 0 -> M)
    extrema[0] = make_float3(0.0, 0.0, 0.0);
    extrema[1] = max; 

    h = 1.4 * ((max.x + max.y + max.z) / 3.0) / particles_along_axis;
    min = std::min(std::min(max.x, max.y), max.z);
    spacing = min / static_cast<float>(particles_along_axis);
  }
};

// identical constants for cpu and gpu access;
Domain h_domain; 
__constant__ Domain domain;

void set_domain_constant(const float3& max, const size_t& particles_along_axis = 50)
{
  h_domain = Domain(max, particles_along_axis);
  // Copy constructed object to device constant memory 
  cudaMemcpyToSymbol(domain, &h_domain, sizeof(Domain));
}

// Spatial lookup creation sort (thrust's sort algos) etc.
struct LookupTable {
  // Local struct for single entry 
  struct entry {
    uint32_t cell_key;
    uint32_t idx; 
  };
  
  // Data 
  entry* data;    // N particles 
  uint32_t* start; 
  uint32_t* end;  // N cells 
};

static size_t calculate_cell_count()
{
  const float3 L = h_domain.extrema[1];
  const size_t x_ct = (L.x / h_domain.h);
  const size_t y_ct = (L.y / h_domain.h);
  const size_t z_ct = (L.z / h_domain.h);
  return x_ct * y_ct * z_ct;
}

__host__ LookupTable create_table(ParticleMatrix* A)
{
  const size_t cell_count = calculate_cell_count();
}

__host__ void update_table(LookupTable* t, ParticleMatrix* A)
{

}

// normal determined by caller 
__device__ float repulsive_force(float r)
{
  const float tol = 1e-2;
  const float k = 1e2;
  
  return -k * (tol - r) / r; 
}

// Update system forces for single timestep 
__global__ void system_forces(ParticleMatrix* A)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx > A->cols) return;  

  // Set d_min and projection as local constants 
  const float d_min = 0.005; 
  const float3 projection = A->x[idx];

  // Check kernel specific particle against its projection onto assumed nearest plane in x,y,z direction and determine force in that direction 

  // proj in X dir 
  if (projection.x < d_min || projection.x - domain.extrema[1].x < d_min)
  {
    // Handle force summation 
    float sign = (projection.x < d_min) ? 1.0 : -1.0;           // Set normal 
    A->fsys[idx].x += (repulsive_force(projection.x) * sign);   // Adjust sign to match normal 
  }

  // Y dir, etc. 
  if (projection.y < d_min || projection.y - domain.extrema[1].y < d_min)
  {
    float sign = (projection.y < d_min) ? 1.0 : -1.0;
    A->fsys[idx].y += (repulsive_force(projection.y) * sign);
  }

  if (projection.z < d_min || projection.z - domain.extrema[1].z < d_min)
  {
    float sign = (projection.z < d_min) ? 1.0 : -1.0;
    A->fsys[idx].z += (repulsive_force(projection.z) * sign);
  }

  // Sum fg
  A->fsys[idx].y += -9.81 * A->mass; //; 
}

// Call at start of program 
__global__ void initial_rng(curandStatePhilox4_32_10_t* s, const size_t seed)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x; 
  curand_init(seed, idx, 0, &s[idx]);
}

// Call reset accumulators kernel after(? or just call reset at start of each loop iter)
__global__ void initialize_particles(curandStatePhilox4_32_10_t* s, ParticleMatrix* A)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > A->cols) return; 

  A->x[idx].x = domain.min * curand_uniform(&s[idx]); 
  A->x[idx].y = domain.min * curand_uniform(&s[idx]); 
  A->x[idx].z = domain.min * curand_uniform(&s[idx]); 

  // Set to zero
  A->v[idx].x = 0.0; 
  A->v[idx].y = 0.0;
  A->v[idx].z = 0.0;
}

__global__ void reset_accumulators(ParticleMatrix* A)
{
  const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > A->cols) return; 

  const float3 zero = make_float3(0.0, 0.0, 0.0);
  // Set values to zero 
  A->fp[idx]   = zero;
  A->fvi[idx]  = zero;
  A->fsys[idx] = zero;
  A->rho[idx]  = 0.0; 
}

__host__ void simulation_loop(ParticleMatrix* A/*LookupBullshit* table*/)
{
  const size_t threads = 256; 
  const size_t particle_grid = (A->cols + 255) / 256;

  reset_accumulators<<<particle_grid, threads>>>(A);
  cudaDeviceSynchronize();

  // First integrate kernel call 


  // Spatial lookup rebuild bullshit 


  // Neighbor search using lookup bullshit  
  

  // Sum up forces 

  
  // Second integrate kernel call 

}
