#include "../include/math_example.hpp"

// for sort test 
#include <algorithm>

#define debug_

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
#pragma unroll 16
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

// Simple comparator to pass to std::sort that keeps track of comparison count 
template <typename T>
class ComparisonCounter {
private:
  size_t comparisons_;
public: 
  ComparisonCounter() : comparisons_(0) {}
  
  bool operator()(const T& a, const T& b) {
    comparisons_++;
    return a < b;
  }

  size_t get_count() const { return comparisons_; }
};

// Wrapper function to benchmark and extract comparison count from std::sort
__attribute__((optnone))
size_t sort_wrapper(float* x, size_t n) {
  ComparisonCounter<float> counter;
#ifdef debug_ 
  std::cout << "Before Sort:\n";
  for (size_t i = 0; i < std::min(size_t(5), n); i++) {
    std::cout << x[i] << '\n';
  }
#endif 
  std::sort(x, x + n, counter);
#ifdef debug_ 
  std::cout << "After Sort:\n";
  for (size_t i = 0; i < std::min(size_t(5), n); i++) {
    std::cout << x[i] << '\n';
  }
#endif 
  return counter.get_count();
}

int main(void) {
  if (avx512_check() == false) {
    std::cout << "No avx512 support!\n";
    return 1;
  }

  float* x = static_cast<float*>(aligned_alloc(64, 16 * sizeof(float)));
  init_array(x, 16);

  auto sort_error_function = [](int a, int b) { return b; };
  Benchmark<float, int, float*, size_t> small_sort_benchmark(sort_error_function, sort_wrapper, 100, x, 16);

  small_sort_benchmark.insert(merge_sort, "ASM Merge Sort");

  init_array(x, 16);
  small_sort_benchmark.run();

  std::cout << "\n " << 16 << " Array Sort:\n\n";

  small_sort_benchmark.get_results();

  auto error_function = [](float base, float result) { return (base - result); };
  Benchmark<float, float, float*> simple_benchmark(error_function, naive_simple_sum, 10000, x);

  std::cout << "\n 16 Element Array Case:\n\n";

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

  // TODO: Create alternate class setup for void types that handle more complex error machanics
  // Example: Two functions to create data structure. Compare time to create. Or whether they initialize to baseline 
  Benchmark<float, size_t, float*, size_t> large_sort_benchmark(sort_error_function, sort_wrapper, 100, x, n*16);

  large_sort_benchmark.insert(merge_sort, "ASM Merge Sort");

  init_array(x, n*16);
  large_sort_benchmark.run();

  std::cout << "\n " << 16*n << " Array Sort:\n\n";

  large_sort_benchmark.get_results();

  Benchmark<float, float, float*, size_t> complex_benchmark(error_function, naive_sum, 10000, x, n);

  std::cout << "\n " << 16*n << " Array Case:\n\n";

  // Compute benchmarks for quicksums 
  complex_benchmark.insert(quicksum_havx512, "Horizontal AVX-512");
  complex_benchmark.insert(quicksum_vavx512, "Vertical   AVX-512");
  complex_benchmark.insert(quicksum_havx2, "Horizontal AVX2");
  complex_benchmark.insert(quicksum_vavx2, "Vertical   AVX2");
  complex_benchmark.insert(quicksum_havx, "Horizontal AVX");
  complex_benchmark.insert(quicksum_vavx, "Vertical   AVX");

  complex_benchmark.run();

  complex_benchmark.get_results();


  free(x);

  return 0;
}
