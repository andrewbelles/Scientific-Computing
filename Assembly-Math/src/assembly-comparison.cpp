#include "../include/math_example.hpp"

// for sort test 
#include <algorithm>
#include <stdexcept>
#include <fstream>

// #define debug_

// Initialize array to be summed 
static void init_array(float* x, int n) {
  unsigned int seed = time(NULL);
  std::uniform_real_distribution<> d {-10.0, 10.0};
  std::default_random_engine rng {seed};

  // Initialize to random real value in range
  for (int i = 0; i < n; i++) {
    x[i] = d(rng);
  }
}

__attribute__((optnone))
float naive_sum(float*& x, size_t& n) {
  float result = 0.0;
  for (size_t i = 0; i < n; i++) {
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

// Wrapper function to benchmark and extract comparison count from std::sort
__attribute__((optnone))
size_t sort_wrapper(float*& x, size_t& n) {
  size_t counter = 0;

  // Set custom lambda that captures counter variable
  std::sort(x, x + n, 
    [&counter](const float& a, const float& b) -> bool
    {
      counter++;
      return a < b;
    }
  );
  return counter;
}

int main(void) {
  // Check if device is avx512 capable
  bool avx512{true};
  if (avx512_check() == false) {
    std::cout << "No avx512 support!\n";
    avx512 = false;
  }

  std::ofstream file("results.txt");
  if (!file.is_open())
  {
    throw::std::runtime_error("File cannot be created");
    return 1;
  }

  size_t size{16};
  float* x = static_cast<float*>(aligned_alloc(64, size * sizeof(float)));
  init_array(x, 16);

  // Returns the difference in comparisons between the 
  auto sort_error_function = [](size_t a, size_t b) -> int64_t { 
    return static_cast<int64_t>(b) - static_cast<int64_t>(a); 
  };
  Benchmark<int64_t, size_t, float*, size_t> small_sort_benchmark(sort_error_function, sort_wrapper, 1000, x, size);

  small_sort_benchmark.insert(merge_sort, "ASM Merge Sort");
  small_sort_benchmark.run();
  small_sort_benchmark.print(file);

  auto error = [](float base, float result) -> float { return result - base; };
  Benchmark<float, float, float*, size_t> simple_benchmark(error, naive_sum, 100000, x, size);

  if (avx512)
  {
    simple_benchmark.insert(sum_horizontal_avx512, "Horizontal AVX-512");
    simple_benchmark.insert(sum_vertical_avx512,   "Vertical   AVX-512");
  }
  simple_benchmark.insert(sum_horizontal_avx2, "Horizontal AVX2");
  simple_benchmark.insert(sum_vertical_avx2,   "Vertical   AVX2");
  simple_benchmark.insert(sum_horizontal_avx, "Horizontal AVX");
  simple_benchmark.insert(sum_vertical_avx,   "Vertical   AVX");
  simple_benchmark.run();
  simple_benchmark.print(file);
  
  free(x);

  size = 16*4096;
  x = static_cast<float*>(aligned_alloc(64, size * sizeof(float)));
  init_array(x, size);

  Benchmark<int64_t, size_t, float*, size_t> large_sort_benchmark(sort_error_function, sort_wrapper, 100, x, size);

  large_sort_benchmark.insert(merge_sort, "ASM Merge Sort");
  large_sort_benchmark.run();
  large_sort_benchmark.print(file);

  Benchmark<float, float, float*, size_t> complex_benchmark(error, naive_sum, 100000, x, size);

  // Compute benchmarks for quicksums 
  if (avx512)
  {
    complex_benchmark.insert(quicksum_havx512, "Horizontal AVX-512");
    complex_benchmark.insert(quicksum_vavx512, "Vertical   AVX-512");
  }
  complex_benchmark.insert(quicksum_havx2, "Horizontal AVX2");
  complex_benchmark.insert(quicksum_vavx2, "Vertical   AVX2");
  complex_benchmark.insert(quicksum_havx, "Horizontal AVX");
  complex_benchmark.insert(quicksum_vavx, "Vertical   AVX");
  complex_benchmark.run();
  complex_benchmark.print(file);

  free(x); // aligned alloc requires free!

  return 0;
}
