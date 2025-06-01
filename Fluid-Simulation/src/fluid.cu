// 2D simulation
#include <GL/glew.h>

// Cuda headers
#include <GL/glext.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

// C++ headers
#include <driver_types.h>
#include <iostream> 
#include <sstream>
#include <fstream>
#include <random>

// External libraries 
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

// Graphics libraries 
#include <raylib.h>

// CPU Constants
constexpr size_t BLOCKSIZE      = 256;  
constexpr float L_host          = 5.0; 
constexpr float rho0_host       = 1.0;
constexpr float2 zero_vec_host{0.0, 0.0};
constexpr size_t MN_host        = 125;
constexpr float c0_host         = 7.0; 
constexpr float visc_host       = 1e-1;

// Matching GPU Constants 
__constant__ size_t MN          = 100;
__constant__ float L            = 5.0;
__constant__ float rho0         = 1.0; 
__constant__ float c0           = 7.0;
__constant__ float visc         = 1e-1;
__constant__ float2 zero_vector{0.0, 0.0};

extern "C" {
__constant__ float poly_C; 
__constant__ float spiky_C;
__constant__ float cubic_C; 
__constant__ float pres_floor;
}

__device__ __host__ float2 add_float2(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ float2 subtract_float2(float2 a, float2 b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}

struct ParticleMatrix 
{
  std::size_t cols; 
  float mass, h, *density;
  float2 *x, *v, *a; 
  float2 *fpres, *fvisc, *fsys;
};


// Macro for all Cuda API Calls to return error and function name etc. of offender 
#define CUDA_CHECK(call)                              \
    do {                                              \
        cudaError_t err__ = (call);                   \
        if (err__ != cudaSuccess) {                   \
            std::ostringstream ss;                    \
            ss << __FILE__ << ':' << __LINE__ << "  " \
               << cudaGetErrorName(err__) << " – "    \
               << cudaGetErrorString(err__);          \
            throw std::runtime_error(ss.str());       \
        }                                             \
    } while (0)


// Spatial lookup structure 
struct Spatial 
{
  struct Value 
  {
    uint2 cell_id; 
    uint64_t key;
    size_t pidx; 
  }; 

  // col number of entries 
  uint2 cells;
  Value *entries;
  size_t *start, *end;
};


// All metadata from device ptrs needed by CPU 
struct Metadata
{
  uint2 cells;
  size_t N; 
  float h;
};


// Compute absolute position in grid space and return to kernel
__device__ uint2 position_to_cell_id(float2 position, float smoothing_radius)
{
  uint2 cell_id;
  cell_id.x = std::floor(position.x / smoothing_radius);
  cell_id.y = std::floor(position.y / smoothing_radius);
  
  float position_y_rel = position.y / smoothing_radius; 

  if (position_y_rel < 0.0)
    printf("Negative Cell ID: (%f, %f) (%u,%u)\n", position.x, position.y, cell_id.x, cell_id.y);

  return cell_id; 
}


// Return a packed key from a key value pair. 
__host__ __device__ uint64_t pack_key(uint2 cell_id, size_t xcells) 
{
  return static_cast<uint64_t>(cell_id.x) * xcells + static_cast<uint64_t>(cell_id.y);
}


// Kernel to compute all particles cell_id 
__global__ void set_keys(ParticleMatrix* particles, const Metadata meta, Spatial::Value* entries, size_t* start, size_t* end)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return; 

  // Get position in register as const 
  const float2 position = particles->x[idx]; 
  const uint2 cell_id   = position_to_cell_id(position, particles->h);

  // Assign values to table 
  entries[idx].cell_id = cell_id;  // TODO: Needed? 
  entries[idx].key     = pack_key(cell_id, meta.cells.x);
  entries[idx].pidx    = idx; 
}


// Get the start and end arrays from key values generated 
__global__ void define_cell_ranges(Spatial::Value* entries, size_t* start, size_t* end, size_t N, size_t xcell)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= N)
    return; 

  const uint64_t key = entries[idx].key;

  // Set idx of start and end cell if non-matching to cell previous to it
  if (idx == 0 || key != entries[idx - 1].key)
    start[key] = idx;

  if (idx + 1 == N || key != entries[idx + 1].key)
    end[key] = idx + 1;
}


// CPU program to generate a spatial table given the current state of particles 
__host__ void generate_spatial_table(ParticleMatrix* particles, Spatial table, const Metadata meta)
{
  const size_t N = meta.N; 
  size_t C = meta.cells.x * meta.cells.y;
  cudaMemset(table.start, 0, C * sizeof(size_t));
  cudaMemset(table.end,   0, C * sizeof(size_t));
  // Kernel call to set cell_id 
  const size_t GRIDSIZE = (N + BLOCKSIZE - 1) / BLOCKSIZE; 
  set_keys<<<GRIDSIZE, BLOCKSIZE>>>(particles, meta, table.entries, table.start, table.end); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Sort entries
  thrust::device_ptr<Spatial::Value> table_entries(table.entries);
  // Lambda comparator for entries with priority to x coordinate then y, then pidx 
  thrust::sort(table_entries, table_entries + N, 
    [] __device__ (const Spatial::Value& a, const Spatial::Value& b)
    {
      return a.key < b.key;
    }
  );

  // Kernel call to set start and end cells 
  define_cell_ranges<<<GRIDSIZE, BLOCKSIZE>>>(table.entries, table.start, table.end, N, meta.cells.x); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


// Cell-block tiling algorithm to generate neighbor list
__global__ void neighbor_search(
    ParticleMatrix* particles, 
    const Metadata meta, 
    Spatial::Value *entries, 
    size_t* start,
    size_t* end, 
    int* neighbor_counts,
    int* neighbor_list)
{
  const size_t cx   = blockIdx.x;
  const size_t cy   = blockIdx.y;
  const size_t cidx = cy * gridDim.x + cx; 

  // Size as defined by grid size 
  extern __shared__ size_t shared_pidx[];
  // Start and end indices 
  const size_t start_index  = start[cidx];     // TODO: Are we certain this is in bounds of start/end 
  const size_t end_index    = end[cidx];

  // Local count of particles 
  const size_t tidx   = threadIdx.x; 
  size_t lcount  = (end_index > start_index) ? (end_index - start_index) : 0;

  // Exit block for invalid lcount 
  __syncthreads();

  if (lcount == 0)
    return; 

  for (size_t base = 0; base < lcount; base += blockDim.x)
  {
    size_t lpid = base + tidx;
    if (lpid < lcount)
      shared_pidx[tidx] = entries[start_index + lpid].pidx;
    __syncthreads();

    if (lpid < lcount) 
    {
        
      // Get position of center particle at idx 
      size_t idx = shared_pidx[tidx];
      float2 x   = particles->x[idx];  

      for (int dy = -1; dy <= 1; dy++)
      {
        // Relative y coordinate, check if in bounds 
        int yrel = cy + dy;
        if (yrel < 0 || yrel >= meta.cells.y)
          continue;

        for (int dx = -1; dx <= 1; dx++)
        {
          // Relative x coordinate
          int xrel = cx + dx;
          if (xrel < 0 || xrel >= meta.cells.x)
            continue;

          // Get relative cell from relative x and y coordinates 
          int rel_cell = xrel + yrel * meta.cells.x;
          size_t rel_start = start[rel_cell];
          size_t rel_end   = end[rel_cell];

          for (size_t p = rel_start; p < rel_end; p++)
          {
            int jdx = entries[p].pidx; 
            // Skip self
            //if (jdx == idx)
            //  continue;

            // relative position 
            float2 xj = particles->x[jdx];
            float2 d = make_float2(x.x - xj.x, x.y - xj.y);
            // Check within smoothing radius
            if (d.x * d.x + d.y * d.y > particles->h * particles->h)
              continue; 

            // Pointer is one more than current neighbor count 
            int ptr = atomicAdd(&neighbor_counts[idx], 1);
            // Check if less than max neighbor count and set 
            if (ptr < MN)
              neighbor_list[idx * MN + ptr] = jdx;
          }
        }
      }
    }
    __syncthreads();
  }
}


// Builds a neighbor list
__host__ void neighbor_host(ParticleMatrix* particles, Spatial table, const Metadata meta, int* neighbor_counts, int* neighbor_list)
{
  const size_t N = meta.N; 
  cudaMemsetAsync(neighbor_counts, 0, N * sizeof(int), 0);    // Set to 0 agnostic to whether it's already been done
  
  // Call w/ 2D grid 
  dim3 GRID(meta.cells.x, meta.cells.y);

  // Ensure shared memory spreads well for each block 
  constexpr size_t RED_BLOCKSIZE = BLOCKSIZE / 2;
  const size_t shared_memory = RED_BLOCKSIZE * sizeof(size_t);
  neighbor_search<<<GRID, RED_BLOCKSIZE, shared_memory>>>(
    particles, 
    meta, 
    table.entries,
    table.start,
    table.end,
    neighbor_counts,
    neighbor_list
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Smoothing Kernel functions 

// poly6 2d density kernel
// Since r isn't explicitly required, opted to only use the squared value of r 
__device__ float poly6(float sqr, float sqh)
{
  float sqd = sqh - sqr;
  // Outside influence - shouldn't occur
  if (sqd <= 0.0)
    return 0.0;

  return poly_C * sqd * sqd * sqd;
}


// Gradient of spiky kernel
__device__ float2 spiky_gradient(float2 r, float h) {
    float rlen = sqrtf(r.x*r.x + r.y*r.y);
    if (rlen == 0.0 || rlen > h) 
      return zero_vector;

    float t = (h - rlen);
    float coeff = -spiky_C * t*t / rlen;  
    return make_float2(r.x * coeff,
                       r.y * coeff);
}


// Laplacian of cubic spline smoothing kernel
__device__ float cubic_spline_laplacian(float2 r, float h)
{
    float rlen = sqrtf(r.x*r.x + r.y*r.y);
    if (rlen > h) 
      return 0.0;

    // 2D constant: 40/(π h^5)
    return cubic_C * (h - rlen);
}


// a tiny kernel, called once per frame:
__global__ void enforce_boundaries(ParticleMatrix* particles) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= particles->cols) return;

  float2 x = particles->x[idx];
  float2 v = particles->v[idx];
  const float restitution = 0.98;

  // left/right
  if (x.x < 1e-3) 
  { 
    x.x =  1e-3;
    v.x =  std::abs(v.x) * restitution; 
  }
  else if (x.x > L - (1e-3))
  { 
    x.x = L - 1e-3;
    v.x = -std::abs(v.x) * restitution; 
  }
  
  // bottom/top
  if (x.y < 1e-3) 
  {
    x.y =  1e-3;
    v.y =  std::abs(v.y) * restitution; 
  }
  else if (x.y > L - (1e-3))
  { 
    x.y = L - (1e-3);
    v.y = -std::abs(v.y) * restitution; 
  }

  particles->x[idx] = x;
  particles->v[idx] = v;
}


// Computes all accumulated density for each particle 
__global__ void compute_densities(ParticleMatrix* particles, const int* neighbor_counts, const int* neighbor_list)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return;

  if (neighbor_counts[idx] == 0)
    printf("Particle %lu has an empty neighbor count\n", idx);

  float sum = 1e-4; 
  float sqh = particles->h * particles->h;
  // Iterate over num of neighbors 
  int cap = std::min(neighbor_counts[idx], static_cast<int>(MN));
  for (int num = 0; num < cap; num++)
  {
    // Idx of relative particle from list  
    int jdx   = neighbor_list[idx * MN + num];

    // Get distance 
    float2 d  = make_float2(
      particles->x[idx].x - particles->x[jdx].x,
      particles->x[idx].y - particles->x[jdx].y
    );

    // Compute density from kernel 
    float sqr = d.x * d.x + d.y * d.y; 
    sum += particles->mass * poly6(sqr, sqh);
  }

  // Set accumulated density 
  particles->density[idx] = sum;
}


// Simple function to return non-linear clamped pressure
__device__ float compute_pressure(float rho_i)
{
  const float B = rho0 * c0 * c0 / 7.0;
  float r = rho_i / rho0;
  return (r > 1.0) ? B * ((r*r*r*r*r*r*r) - 1.0) : pres_floor;
}


// Density precomputed - Compute all forces  
__global__ void compute_forces(ParticleMatrix* particles, const int* neighbor_counts, const int* neighbor_list)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return; 

  // Get local values for particle 
  const float2 x      = particles->x[idx];
  const float xrho    = particles->density[idx]; 

  float xpres = compute_pressure(xrho); 

  const float2 v      = particles->v[idx];

  particles->fsys[idx].y += particles->mass * -98100; 
  
  int cap = std::min(neighbor_counts[idx], static_cast<int>(MN));
  for (int num = 0; num < cap; num++)
  {
    // Get relative values for neighbor 
    size_t jdx  = neighbor_list[idx * MN + num];
    float2 xrel = particles->x[jdx];
    float2 dj   = make_float2(x.x - xrel.x, x.y - xrel.y);

    // Compute pressure from density for relative particle
    float jrho = particles->density[jdx];
    float jpres = compute_pressure(jrho);

    // Compute the pressure force 
    float a = -particles->mass * ((xpres / (xrho * xrho)) + (jpres / (jrho * jrho)));
    float2 gW = spiky_gradient(dj, particles->h);
    float2 av = make_float2(gW.x * a, gW.y * a);
    particles->fpres[idx] = add_float2(particles->fpres[idx], av);

    // Compute the viscosity force 
    float laplacian = cubic_spline_laplacian(dj, particles->h);
    float b = visc * particles->mass / jrho * laplacian;
    float2 relv = subtract_float2(particles->v[jdx], v);
    particles->fvisc[idx] = add_float2(particles->fvisc[idx], make_float2(relv.x * b, relv.y * b));
  }

  // Compute acceleration directly
  float2 ftotal = add_float2(particles->fpres[idx], add_float2(particles->fvisc[idx], particles->fsys[idx]));
  particles->a[idx] = make_float2(ftotal.x / xrho, ftotal.y / xrho);
}


// Computes first half of integration with half a timestep 
__global__ void verlet_kick(ParticleMatrix* particles, float half_dt)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return;

  particles->v[idx].x += particles->a[idx].x * half_dt;
  particles->v[idx].y += particles->a[idx].y * half_dt; 
}


// Computes second half after forces have been calculated 
__global__ void verlet_drift(ParticleMatrix* particles, float half_dt)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return;

  particles->x[idx].x += particles->v[idx].x * half_dt;
  particles->x[idx].y += particles->v[idx].y * half_dt;
}


__global__ void reset_accumulators(ParticleMatrix* particles)
{
  const size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  if (idx >= particles->cols)
    return;

  //if (particles->x[idx].x > L || particles->x[idx].x < 0 || particles->x[idx].y > L || particles->x[idx].y < 0)
  //  printf("Particle out of bounds (%f,%f): %lu\n", particles->x[idx].x, particles->x[idx].y, idx);

  // Zero out accumulated values 
  particles->density[idx] = 1e-4;
  particles->fsys[idx] = particles->fpres[idx] = particles->fvisc[idx] = zero_vector; 
}


__host__ float adaptive_dt(ParticleMatrix* particles, const Metadata& meta)
{
  const float CFL    = 0.2; 
  const float CFORCE = 0.25;
  const float CVISC  = 0.125;

  // Reduce the max magnitude from across all particles efficiently 
  float max_velocity = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(static_cast<int>(meta.N)), 
    [=] __device__ (int i) -> float
    {
      float2 v = particles->v[i];
      return std::sqrt(v.x * v.x + v.y * v.y);
    },
    0.0, 
    thrust::maximum<float>()
  );

  float max_accel = thrust::transform_reduce(
    thrust::device,
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(static_cast<int>(meta.N)), 
    [=] __device__ (int i) -> float
    {
      float2 a = particles->a[i];
      return std::sqrt(a.x * a.x + a.y * a.y);
    },
    0.0, 
    thrust::maximum<float>()
  );

  float dt_acoustic = CFL * (meta.h / (c0_host + max_velocity));
  float dt_force    = CFORCE * std::sqrt(meta.h / (max_accel + 1e-3));
  float dt_visc     = CVISC * (meta.h * meta.h / (visc_host + 1e-3));
  
  // return smallest time constant
  return std::min(dt_acoustic, std::min(dt_force, dt_visc)); 
}

// Compute the forces given the filled neighbor counts and list
// Don't need table. 
__host__ void handle_forces(ParticleMatrix* particles, const Metadata& meta, int* neighbor_counts, int* neighbor_list)
{
  // TODO: Compute dynamic timestep 
  const float dt = adaptive_dt(particles, meta);
  const size_t GRIDSIZE = (meta.N + BLOCKSIZE - 1) / BLOCKSIZE;

  // KERNEL CALLS - All use the same launch parameters  
  reset_accumulators<<<GRIDSIZE, BLOCKSIZE>>>(particles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  compute_densities<<<GRIDSIZE, BLOCKSIZE>>>(particles, neighbor_counts, neighbor_list);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  compute_forces<<<GRIDSIZE, BLOCKSIZE>>>(particles, neighbor_counts, neighbor_list);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  verlet_kick<<<GRIDSIZE, BLOCKSIZE>>>(particles, dt/2.0); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  verlet_drift<<<GRIDSIZE, BLOCKSIZE>>>(particles, dt);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  enforce_boundaries<<<GRIDSIZE, BLOCKSIZE>>>(particles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  compute_densities<<<GRIDSIZE, BLOCKSIZE>>>(particles, neighbor_counts, neighbor_list);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  compute_forces<<<GRIDSIZE, BLOCKSIZE>>>(particles, neighbor_counts, neighbor_list);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  verlet_kick<<<GRIDSIZE, BLOCKSIZE>>>(particles, dt/2.0); 
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


namespace bufr {

// Struct to hold buffer 
struct Buffer 
{
  GLuint pos_vbo{0};
  GLuint rho_vbo{0};
  cudaGraphicsResource* pos_res{nullptr};
  cudaGraphicsResource* rho_res{nullptr};
};


// Takes number of particles and sets up buffers
void initialize_cuda_buffers(Buffer* buffers, size_t N)
{
  // Setup buffer for position and allocate resources from cuda 
  glGenBuffers(1, &buffers->pos_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, buffers->pos_vbo);
  glBufferData(GL_ARRAY_BUFFER, N * sizeof(float2), nullptr, GL_DYNAMIC_DRAW); 
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(&buffers->pos_res, buffers->pos_vbo, cudaGraphicsRegisterFlagsNone);

  // And density 
  glGenBuffers(1, &buffers->rho_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, buffers->rho_vbo);
  glBufferData(GL_ARRAY_BUFFER, N * sizeof(float), nullptr, GL_DYNAMIC_DRAW); 
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(&buffers->rho_res, buffers->rho_vbo, cudaGraphicsRegisterFlagsNone);
}


void update_buffers(Buffer* buffers, ParticleMatrix host_particles)
{
  size_t size; 
  float2* d_pos;
  float* d_rho;

  cudaGraphicsMapResources(1, &buffers->pos_res, 0);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_pos), &size, buffers->pos_res);
  cudaMemcpy(d_pos, host_particles.x, host_particles.cols * sizeof(float2), cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, &buffers->pos_res, 0);

  cudaGraphicsMapResources(1, &buffers->rho_res, 0);
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&d_rho), &size, buffers->rho_res);
  cudaMemcpy(d_rho, host_particles.density, host_particles.cols * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaGraphicsUnmapResources(1, &buffers->rho_res, 0);
}


// Free allocated resourcess 
void unregister_buffers(Buffer* buffers)
{
  // Check for nullptr and unregister 
  if (buffers->pos_res != nullptr)
  {
    cudaGraphicsUnregisterResource(buffers->pos_res);
    glDeleteBuffers(1, &buffers->pos_vbo);
    buffers->pos_res = nullptr;
  }

  // Again for density 
  if (buffers->rho_res != nullptr)
  {
    cudaGraphicsUnregisterResource(buffers->rho_res);
    glDeleteBuffers(1, &buffers->rho_vbo);
    buffers->rho_res = nullptr;
  }
}

}

namespace shdr {

// Read all data from shader file into return value 
// Credit ChatGPT o4-mini-high
std::string load_shader(const std::string& path)
{
  std::ifstream shader_file(path, std::ios::in | std::ios::binary);
  if (!shader_file)
    throw std::runtime_error("File doesn't exist");

  std::string shader; 
  shader_file.seekg(0, std::ios::end);
  shader.resize(shader_file.tellg());
  shader_file.seekg(0, std::ios::beg);
  shader_file.read(&shader[0], shader.size());
  shader_file.close();

  return shader;
}


// Compile the shader
GLuint compile_shader(GLenum type, const std::string& src)
{
  char error[512];
  GLint status;
  GLuint shader = glCreateShader(type);
  const char* cstr = src.c_str();

  glShaderSource(shader, 1, &cstr, nullptr);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status)
  {
    glGetShaderInfoLog(shader, 512, nullptr, error); 
    throw std::runtime_error(error);
  }

  return shader;
}


// Creates the program from file paths to fragment and vertex shaders 
GLuint create_program(const std::string& fragment_path, const std::string& vertex_path)
{
  // Get shaders from source and compile 
  std::string vertex_src   = load_shader(vertex_path);
  std::string fragment_src = load_shader(fragment_path);
  GLuint vertex_shader   = compile_shader(GL_VERTEX_SHADER, vertex_src);
  GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_src);
  GLuint program = glCreateProgram();

  GLint status;
  char error[512];

  // Attach shaders to program 
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);

  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (!status)
  {
    glGetProgramInfoLog(program, 512, nullptr, error);
    glDeleteProgram(program);
    throw std::runtime_error(error);
  }

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return program;
}

}

// Main 
int main(void)
{
  // Create opengl context 
  InitWindow(800, 800, "SPH");
  glewInit();
  glEnable(GL_PROGRAM_POINT_SIZE);

  GLuint program = shdr::create_program("../shaders/fragment.frag", "../shaders/vertex.vert");
  GLint projection_location = glGetUniformLocation(program, "uProj");

  GLuint point_vao;
  glGenVertexArrays(1, &point_vao);
  glBindVertexArray(point_vao);

  // System constants
  constexpr size_t N = 8000; 

  constexpr float region_size = 2.5;
  constexpr float offset      = (L_host - region_size) / 2.0;

  constexpr float area = L_host * L_host / 2.0;
  constexpr float mass = rho0_host * area;
  constexpr float mass_per = mass / static_cast<float>(N);

  //const size_t Mx = static_cast<size_t>(std::ceil(std::sqrt(2.0 * static_cast<float>(N))));
  const size_t M    = static_cast<size_t>(std::ceil(std::sqrt(static_cast<float>(N))));
  const float delta = region_size / static_cast<float>(M);
  //const size_t My = static_cast<size_t>(std::ceil((L_host / 2.0) / delta));

  const float h = 1.3 * delta; 
  const float r = h * 0.8 * (600 / L_host);

  // Set metadata
  Metadata meta = (Metadata)
  {
    .cells = make_uint2(std::ceil(L_host / h), std::ceil(L_host / h)),
    .N     = N,
    .h     = h
  };

  // Allocate memory to particle device ptr
  ParticleMatrix h_p, *particles;
  cudaMalloc(&h_p.density, N * sizeof(float));
  cudaMalloc(&h_p.x, N * sizeof(float2)); 
  cudaMalloc(&h_p.v, N * sizeof(float2));
  cudaMalloc(&h_p.a, N * sizeof(float2));
  cudaMalloc(&h_p.fpres, N * sizeof(float2)); 
  cudaMalloc(&h_p.fvisc, N * sizeof(float2));
  cudaMalloc(&h_p.fsys, N * sizeof(float2));
  cudaMalloc(&particles, sizeof(ParticleMatrix));


  // Set constant values 
  h_p.cols = N;
  h_p.mass = mass_per; 
  h_p.h    = h;
  
  // Particle position initialization
  
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> drift(-0.1f*delta, 0.1f*delta);

  std::vector<float2> host_positions;
  std::vector<float2> host_velocities(N, zero_vec_host);
  std::vector<float2> host_accelerations(N, zero_vec_host);

  host_positions.reserve(N);

  int pid = 0;
  for (size_t i = 0; i < M && pid < N; ++i) 
  {
    for (size_t j = 0; j < M && pid < N; ++j) 
    {
      float x = offset + (i + 0.5) * delta + drift(rng);
      float y = offset + (j + 0.5) * delta + drift(rng);

      // Clamp positions
      if (x < offset) 
        x = offset + 1e-4;
      else if (x > offset + region_size) 
        x = offset + region_size - 1e-4;

      if (y < offset) 
        y = offset + 1e-4;
      else if (y > offset + region_size) 
        y = offset + region_size - 1e-4;
      
      host_positions.emplace_back(make_float2(x, y));
      pid++;
    }
  }

  // Copy initialized vectors to device  
  cudaMemcpy(h_p.x, host_positions.data(), N * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(h_p.v, host_velocities.data(), N * sizeof(float2), cudaMemcpyHostToDevice);
  cudaMemcpy(h_p.a, host_accelerations.data(), N * sizeof(float2), cudaMemcpyHostToDevice);

  // Copy all to device pointer 
  cudaMemcpy(particles, &h_p, sizeof(ParticleMatrix), cudaMemcpyHostToDevice);

  // Table is already a host pointer so no memcpy required to a device pointer
  Spatial table;
  table.cells = meta.cells; 
  cudaMalloc(&table.entries, N * sizeof(Spatial::Value));
  cudaMalloc(&table.start, meta.cells.x * meta.cells.y * sizeof(size_t));
  cudaMalloc(&table.end, meta.cells.x * meta.cells.y * sizeof(size_t));

  // Allocate memory for lists
  int *neighbor_list, *neighbor_counts; 
  cudaMalloc(&neighbor_counts, N * sizeof(int));
  cudaMalloc(&neighbor_list, N * MN_host * sizeof(int));

  // setting up matrix 
  float projection_matrix[16] = {
      2.0/L_host, 0,        0,  0,
      0,          2.0/L_host, 0,  0,
      0,          0,       -1,  0,
     -1.0,       -1.0,     0,   1
  };

  const float h8 = std::pow(h, 8);
  const float h5 = std::pow(h, 5);

  float POLY_CONST  = 4.0  / (M_PI * h8);
  float SPIKY_CONST = 30.0 / (M_PI * h5);
  float CUBIC_CONST = 40.0 / (M_PI * h5);
  constexpr float pres_floor_host = 0.01 * rho0_host * c0_host * c0_host;

  CUDA_CHECK(cudaMemcpyToSymbol(poly_C, &POLY_CONST, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(spiky_C, &SPIKY_CONST, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(cubic_C, &CUBIC_CONST, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(pres_floor, &pres_floor_host, sizeof(float)));

  // Set up buffers 
  bufr::Buffer buffer;
  bufr::initialize_cuda_buffers(&buffer, N);

  // Simulation loop
  SetTargetFPS(144);
  while (!WindowShouldClose())
  {

    // One timestep of simulation  
    generate_spatial_table(particles, table, meta);
    neighbor_host(particles, table, meta, neighbor_counts, neighbor_list);
    handle_forces(particles, meta, neighbor_counts, neighbor_list);

    bufr::update_buffers(&buffer, h_p);

    // Handle drawing from buffer 

    BeginDrawing();

      ClearBackground(BLACK);

      // View port should match rectangle sizing 
      DrawRectangleLines(100, 100, 600, 600, WHITE);
      glViewport(100, 100, 600, 600);

      // Get position buffer 
      glBindVertexArray(point_vao);
      glUseProgram(program);
      glUniform1f(glGetUniformLocation(program, "pointSize"), r);
      // Map projection matrix for particle positions 
      glUniformMatrix4fv(projection_location, 1, GL_FALSE, projection_matrix);

      // Call buffers and draw points 
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, buffer.pos_vbo);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
      glDrawArrays(GL_POINTS, 0, N);

      // Disable buffers 
      glDisableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindVertexArray(0);
      glUseProgram(0);
      glViewport(0, 0, 800, 800);

    EndDrawing();
  }

  // Free resources 

  glDeleteProgram(program);
  CloseWindow();
  bufr::unregister_buffers(&buffer);

  cudaFree(neighbor_counts);
  cudaFree(neighbor_list); 
  cudaFree(table.entries);
  cudaFree(table.start);
  cudaFree(table.end);

  // Copy particles back to host to free resources 
  cudaMemcpy(&h_p, particles, sizeof(ParticleMatrix), cudaMemcpyDeviceToHost);

  cudaFree(h_p.x);
  cudaFree(h_p.v);
  cudaFree(h_p.a);
  cudaFree(h_p.density);
  cudaFree(h_p.fsys);
  cudaFree(h_p.fpres);
  cudaFree(h_p.fvisc);
  cudaFree(particles);

  return 0; 
}
