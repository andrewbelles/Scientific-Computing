#include <cuda_runtime.h>
#include <cublas.h>
#include <algorithm>

struct ParticleMatrix {
  // metadata for gpu 
  size_t cols;
  size_t rows;    // Equivalent to the # of stored state information
  
  // Plane boundaries should be managed on gpu (?)

  // elements of matrix 
  float* rho;
  float3* x;
  float3* v;
  float3* fp;
  float3* fvi;
  float3* fsys;
  
  ParticleMatrix(size_t columns, size_t rows) : cols(columns), rows(rows) 
  {
    // Create memory for each particle
    cudaMallocAsync(&rho, cols * sizeof(float), 0);
    cudaMallocAsync(&x, cols * sizeof(float3), 0);
    cudaMallocAsync(&v, cols * sizeof(float3), 0);
    cudaMallocAsync(&fp, cols * sizeof(float3), 0);
    cudaMallocAsync(&fvi, cols * sizeof(float3), 0);
    cudaMallocAsync(&fsys, cols * sizeof(float3), 0);
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

// Symbolically linked Matrix struct that contains the metadata of a gpu managed structure
enum class Operation : uint8_t { None, Transpose, Conjugate };
enum class Layout : uint8_t { Row, Col };

struct MatrixData
{
  size_t cols;
  size_t rows;
  size_t lda; 
  cudaStream_t stream{0};   // Maybe? 
  
  // Enums 
  Layout layout;
  Operation op;

  MatrixData(size_t cols, size_t rows, cudaStream_t stream = 0) 
    : cols(cols), rows(rows), stream(stream) {}
};

namespace dmn 
{

// Define the domain under which particles will be initialized etc 
class Domain 
{
public: 

  float3 extrema[2];
  float3 normals[6]; 
  float spacing{0.0};
  float length{0.0}; 

  Domain(float3 max, size_t particles_along_axis = 50)
  {
    // Extrema values of plane (from 0 -> M)
    extrema[0] = make_float3(0.0, 0.0, 0.0);
    extrema[1] = max; 

    // Set normal vectors 
    normals[0] = make_float3(1.0, 0.0, 0.0);
    normals[1] = make_float3(-1.0, 0.0, 0.0);
    normals[2] = make_float3(0.0, 1.0, 0.0);
    normals[3] = make_float3(0.0, -1.0, 0.0);
    normals[4] = make_float3(0.0, 0.0, 1.0);
    normals[5] = make_float3(0.0, 0.0, -1.0);

    spacing = std::min(std::min(max.x, max.y), max.z) / static_cast<float>(particles_along_axis);
  }

  static __global__ void system_forces(ParticleMatrix* A)
  {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x; 
  }

};

}
