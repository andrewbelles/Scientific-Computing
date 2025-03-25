#include "../benchmark.hpp"

__attribute__((optnone))
static float add(float a, float b)
{
 return a + b;
}

__attribute__((optnone))
static float add_by_ref(float& a, float& b)
{
  return a + b;
}

// Awful addition of floats function
__attribute__((optnone))
static float bad_add(float a, float b)
{
  // Take integer portion
  int a_r = static_cast<int>(a);
  int b_r = static_cast<int>(b);

  // Take decimal 
  float a_d = a - static_cast<float>(a_r);
  float b_d = b - static_cast<float>(b_r);

  // Introduce some faux error
  a_d -= 0.01;
  
  return (a_d + b_d) + (static_cast<float>(a_r) + static_cast<float>(b_r));
}

// Reference impl
__attribute__((optnone))
static float bad_add_by_ref(float& a, float& b)
{
  // Take integer portion
  int a_r = static_cast<int>(a);
  int b_r = static_cast<int>(b);

  // Take decimal 
  float a_d = a - static_cast<float>(a_r);
  float b_d = b - static_cast<float>(b_r);

  // Introduce some faux error
  a_d -= 0.01;
  
  return (a_d + b_d) + (static_cast<float>(a_r) + static_cast<float>(b_r));
}



int main(void)
{
  float a = 3.4, b = 4.3;

  auto error = [](float a, float b) { return a - b; };
  Benchmark<float, float, float, float> b_ref(error, add_by_ref, 100, a, b); 
  
  // This will cause the program to fail to compile as the add function does not pass by reference
  //Benchmark<float, float, float, float> b_noref(error, add, 100, a, b);

  b_ref.print();

  b_ref.insert(bad_add_by_ref, "Awful Impl");
  //b_ref.insert(bad_add, "Compile Error");

  b_ref.run();

  b_ref.print();

  return 0;
}
