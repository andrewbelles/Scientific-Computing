#include "../benchmark.hpp"
#include <cmath>
#include <iostream>

// Simple class implementation to show that benchmark works for class methods

class Example
{
public:
  float a, b;

  Example(float in, float out) : a(in), b(out), a_initial(in), b_initial(out) {}

public:
  
  // What the benchmark interfaces with 
  float base_interface(float& divisor) 
  {
    a = a_initial;
    b = b_initial;

    a /= divisor;
    impure_function();
    return b;
  }

  float test_interface(float& divisor)
  {
    a = a_initial;
    b = b_initial;

    a /= divisor;
    return b;
  }

private:
  float a_initial, b_initial;

  void impure_function()
  {
    b = std::pow(a, b);
  }
};

__attribute__((optnone))
float loop(float& dummy)
{
  (void)dummy;
  float result = 0.0;
  for (int i = 0; i < 100; i++)
  {
    result += 1.0;
  }
  return result;
}

int main(void)
{
  Example c(100.0, 2.0);

  auto error = [](float a, float b) { return a - b; };
  auto bench = [&c](float& divisor) { return c.base_interface(divisor); };
  Benchmark<float, float, float> b(error, bench, 1000000, 5.0);

  auto non_naive = [&c](float& divisor) { return c.test_interface(divisor); };
  b.insert(non_naive, "Pure Interface");
  b.run();
  b.print();

  std::cout << "\n\n";

  b.insert(loop, "Slow function outside Class");
  b.run();
  b.print();

  return 0;
}
