#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>

#define ARR_SIZE 10240
#define WORK_ITERATIONS_LE 99000
#define WORK_ITERATIONS_GT 100100

// Print an error message and exit with failure code
#define DIE(...) \
{ \
        fprintf(stderr, __VA_ARGS__); \
        fflush(stderr); \
        exit(EXIT_FAILURE); \
}

#define cudaAssert(err) cudaCheckError((err), __FILE__, __LINE__)
void cudaCheckError(cudaError_t err, const char *file, int line);

// A factorial function. Implemented [almost] the worst way possible.
__device__
int factorial(int n);

// A step in the work for x <= 0.5
// [0, 0.5] -> [0, 0.5]
// It [almost] follows this function: https://bit.ly/2FjdsL0
__device__
double next_step_le_half(double x);

// A step in the work for x > 0.5
// ]0.5, 1] -> ]0.5, 1]
// It [almost] follows this function: https://bit.ly/31D982B
__device__
double next_step_gt_half(double x);

// The two versions of the work. The second is expected to be faster.
// Both should receive the work at "arr" and return at "results".
void launch_gpu_work_v1(double *arr, double **results);
void launch_gpu_work_v2(double *arr, double **results);

#endif
