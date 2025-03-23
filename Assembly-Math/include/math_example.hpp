#ifndef MATH_EXAMPLE
#define MATH_EXAMPLE 

#include <iostream>
#include <random>
// #include <benchmark.hpp>
#include "/home/andrew/Repositories/miscellaneous/benchmark.hpp"

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
extern "C" float quicksum_havx2(float* x, size_t n);
extern "C" float quicksum_vavx2(float* x, size_t n);
extern "C" float quicksum_havx(float* x, size_t n);
extern "C" float quicksum_vavx(float* x, size_t n);

// Assembly Sort Functions 
extern "C" int merge_sort(float* x, size_t n);

#endif // MATH_EXAMPLE
