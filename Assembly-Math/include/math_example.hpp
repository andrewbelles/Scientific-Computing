#ifndef MATH_EXAMPLE
#define MATH_EXAMPLE 

#include <iostream>
#include <random>
#include "/home/andrew/Repositories/Scientific-Computing/Benchmark-Tool/benchmark.hpp"

// Check for avx512 support
#include <immintrin.h>

// Simple ASM Reductions 
extern "C" float sum_horizontal_avx512(float*& x, size_t& n);
extern "C" float sum_vertical_avx512(float*& x, size_t& n);
extern "C" float sum_horizontal_avx2(float*& x, size_t& n);
extern "C" float sum_vertical_avx2(float*& x, size_t& n);
extern "C" float sum_horizontal_avx(float*& x, size_t& n);
extern "C" float sum_vertical_avx(float*& x, size_t& n);

// Advanced Reductions 
extern "C" float quicksum_havx512(float*& x, size_t& n);
extern "C" float quicksum_vavx512(float*& x, size_t& n);
extern "C" float quicksum_havx2(float*& x, size_t& n);
extern "C" float quicksum_vavx2(float*& x, size_t& n);
extern "C" float quicksum_havx(float*& x, size_t& n);
extern "C" float quicksum_vavx(float*& x, size_t& n);

// Assembly Sort Functions 
extern "C" size_t merge_sort(float*& x, size_t& n);

#endif // MATH_EXAMPLE
