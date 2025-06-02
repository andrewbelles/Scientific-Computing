#include "../benchmark.hpp"
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector> 
#include <iostream>

// rng 
static std::default_random_engine rng(std::random_device{}());
static std::uniform_real_distribution<> d(0.0, 1000000.0);

// Returns some random value in large range to operate on
static float random_float()
{
  return d(rng);
}

// Produce vector of random floats 
static std::vector<float> random_vector_float(size_t size)
{
  std::vector<float> vec(size);
  for (size_t i = 0; i < size; i++)
  {
    vec[i] = random_float();
  }
  return vec;
}

//__attribute__((optnone))
static float naive_square_root(float& x)
{
  float guess = x / 2.0;
  for (int i = 0; i < 15; i++)
  {
    guess = 0.5 * (guess + x / guess);
  }
  return guess;
}

// Simple embedded sqrt 
static float emb_sqrt(float& x)
{
  float res;
  asm
  (
    "sqrtss %1, %0"
    : "=x" (res)
    : "x" (x)
  );
  return res;
}

// __attribute__((optnone))
static float sqrt_wrapper(float& x)
{
  return std::sqrt(x);
}

// Vector benchmark functions
// __attribute__((optnone))
static float sqrt_vec_wrapper(std::vector<float>& x)
{
  float average = 0.0; 
  for (size_t i = 0; i < x.size(); i++)
  {
    average += std::sqrt(x[i]);
  }
  return average / static_cast<float>(x.size());
}

// __attribute__((optnone))
static float naive_vec_sqrt(std::vector<float>& x)
{
  float average = 0.0; 
  for (size_t i = 0; i < x.size(); i++)
  {
    float guess = x[i] / 2.0;
    for (int j = 0; j < 15; j++)
    {
      guess = 0.5 * (guess + x[i] / guess);
    }
    average += guess;
  }
  return average / static_cast<float>(x.size());
}

template<typename T>
size_t std_sort_wrapper(std::vector<T>& x)
{
  // Pre and post should be different for all iters
  size_t comparisons = 0;
  std::sort(x.begin(), x.end(), [&comparisons](const T& a, const T&b)
  {
    comparisons++;
    return a < b;
  });
  return comparisons;
}

template<typename T>
size_t naive_selection_sort(std::vector<T>& x)
{
  const size_t size = x.size();
  size_t comparisons = 0;
  for (size_t i = 0; i < size; i++)
  {
    for (size_t j = 0; j < size; j++)
    {
      if (i == j) continue;

      comparisons++;
      if (x[i] > x[j]) {
        T temp = x[i];
        x[i]   = x[j];
        x[j]   = temp;
      }
    }
  }
  return comparisons;
}

template<typename T>
size_t std_sort_wrapper_raw(T*& x, size_t& elements)
{

  size_t comparisons = 0;
  std::sort(x, x + elements, [&comparisons](const T& a, const T&b)
  {
    comparisons++;
    return a < b;
  });
  return comparisons;
}

template<typename T>
size_t naive_selection_sort_raw(T*& x, size_t& elements)
{
  size_t comparisons = 0;
  for (size_t i = 0; i < elements; i++)
  {
    for (size_t j = 0; j < elements; j++)
    {
      if (i == j) continue;

      comparisons++;
      if (x[i] > x[j]) {
        T temp = x[i];
        x[i]   = x[j];
        x[j]   = temp;
      }
    }
  }
  return comparisons;
}

/*
typedef struct Node {
  int key;
  struct Node *next; 
} Hashtable;
*/

int main(void)
{
  std::cout << "\nSimple Benchmark Test\n\n";

  float input = random_float();


  /* 

  auto* table = new Hashtable[37];
  auto* base_table = new Hashtable[37];
  
  // void benchmark_function(Hashtable* table, int* values_to_hash, size_t n_elements);
  // size_t error_function(void); 

  auto error_function = [&table, &base_table]() { 
    size_t difference = 0;
    
    for (size_t i = 0; i < 37; i++)
    {
      auto* curr = &table[i];
      auto* base_curr = &base_table[i];
      while (curr != nullptr && base_curr != nullptr)
      {
        if (curr->key != base_curr->key)
        {
          difference++;
        }
        curr = curr->next;
        base_curr = base_curr->next;
      }
    }
    return difference;
  };
  */

  auto error_function = [](float a, float b) { return a - b; };
  Benchmark<float, float, float> simple_benchmark(error_function, sqrt_wrapper, 1000000, input);
  simple_benchmark.insert(naive_square_root, "Newton's Method");
  simple_benchmark.insert(emb_sqrt, "Embedded Assembly");

  simple_benchmark.run();
  simple_benchmark.print();

  std::cout << "\nContainer Benchmark Test\n\n";

  // Initialize to zero
  std::vector<float> vec_input = random_vector_float(4096);

  Benchmark<float, float, std::vector<float>> container_benchmark(error_function, sqrt_vec_wrapper, 1000, vec_input);
  container_benchmark.insert(naive_vec_sqrt, "Newton's Method");

  container_benchmark.run();
  container_benchmark.print();

  std::cout << "\nContainer Sort Test (Copy must be Respected)\n\n";

  // Resuse vec_input 
  // Difference in number of comparisons 
  auto sort_error = [](size_t a, size_t b) { 
    return static_cast<int64_t>(a) - static_cast<int64_t>(b);
  };

  Benchmark<int64_t, size_t, std::vector<float>&> container_sort(sort_error, std_sort_wrapper<float>, 1000, vec_input);

  container_sort.insert(naive_selection_sort<float>, "Selection Sort");

  container_sort.run();
  container_sort.print();

  std::cout << "\nRaw Pointer Sort Test\n\n";

  auto *raw_array = new float[4096];
  std::memcpy(raw_array, vec_input.data(), vec_input.size() * sizeof(float));
  Benchmark<int64_t, size_t, float*, size_t> raw_sort(sort_error, std_sort_wrapper_raw<float>, 1000, raw_array, 4096);

  raw_sort.insert(naive_selection_sort_raw<float>, "Selection Sort");

  raw_sort.run();
  raw_sort.print();

  delete[] raw_array;

  return 0;
}
