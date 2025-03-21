#include <cstdlib>
#include <iostream>
#include <random>
#include <benchmark.hpp>

// Check for avx512 support
#include <immintrin.h>

// Simple ASM Reductions 
extern "C" float sum_horizontal_avx512(float* x);
extern "C" float sum_vertical_avx512(float* x);
extern "C" float sum_horizontal_avx2(float* x);
extern "C" float sum_vertical_avx2(float* x);
extern "C" float sum_horizontal_avx(float* x);
extern "C" float sum_vertical_avx(float* x);

// Advanced Reductions 
extern "C" float quicksum_havx512(float* x, size_t n);
extern "C" float quicksum_vavx512(float* x, size_t n);

// Initialize array to be summed 
static void init_array(float* x, int n) {
  unsigned int seed = time(NULL);
  std::uniform_real_distribution<> d {-5.0, 5.0};
  std::default_random_engine rng {seed};

  // Initialize to random real value in range
  for (int i = 0; i < n; i++) {
    x[i] = d(rng);
  }

}

// Naive sum implementation 
__attribute__((optnone))
static float naive_simple_sum(float* x) {
  float result = 0.0;
  for (int i = 0; i < 16; i++) {
    result += x[i];
  }
  return result;
}

__attribute__((optnone))
static float unrolled_simple_sum(float* x) {
  float result = 0.0;
#pragma unroll 4
  for (int i = 0; i < 16; i++) {
    result += x[i];
  }
  return result;
}

__attribute__((optnone))
static float naive_sum(float* x, size_t n) {
  float result = 0.0;
  for (size_t i = 0; i < 16*n; i++) {
    result += x[i];
  }
  return result;
}

// Simple check for native avx512 on cpu 
bool avx512_check() {
  if (!__builtin_cpu_supports("avx512f"))
    return false;

  return true;
}

int main(void) {
  if (avx512_check() == false) {
    std::cout << "No avx512 support!\n";
    return 1;
  }

  float* x = static_cast<float*>(aligned_alloc(64, 16 * sizeof(float)));
  init_array(x, 16);

  auto error_function = [](float base, float result) { return (base - result); };
  Benchmark<float, float, float*> simple_benchmark(naive_simple_sum, error_function, 100000, x);

  std::cout << "\n16 Element Array Case:\n\n";

  simple_benchmark.insert(unrolled_simple_sum, "Unrolled Naive");
  simple_benchmark.insert(sum_horizontal_avx512, "Horizontal AVX-512");
  simple_benchmark.insert(sum_vertical_avx512,   "Vertical   AVX-512");
  simple_benchmark.insert(sum_horizontal_avx2, "Horizontal AVX2");
  simple_benchmark.insert(sum_vertical_avx2,   "Vertical   AVX2");
  simple_benchmark.insert(sum_horizontal_avx, "Horizontal AVX");
  simple_benchmark.insert(sum_vertical_avx,   "Vertical   AVX");

  simple_benchmark.run();

  simple_benchmark.get_results();
  
  free(x);

  size_t n = 4096;
  x = static_cast<float*>(aligned_alloc(64, 16 * n * sizeof(float)));
  init_array(x, n*16);

  Benchmark<float, float, float*, size_t> complex_benchmark(naive_sum, error_function, 100000, x, n);

  std::cout << "\nLarger Array Case:\n\n";

  // Compute benchmarks for quicksums 
  complex_benchmark.insert(quicksum_havx512, "Horizontal AVX-512");
  complex_benchmark.insert(quicksum_vavx512, "Vertical   AVX-512");

  complex_benchmark.run();

  complex_benchmark.get_results();

  free(x);

  return 0;
}
