#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>

#define ARR_SIZE 1000000
extern __device__ int GPU_WORK_ITERATIONS;

#define DIV_CEIL_INT(x, y) (1 + (((x) - 1) / (y)))

// Print an error message and exit with failure code
#define DIE(...) \
{ \
        fprintf(stderr, __VA_ARGS__); \
        fflush(stderr); \
        exit(EXIT_FAILURE); \
}

#define cudaAssert(err) cudaCheckError((err), __FILE__, __LINE__)
void cudaCheckError(cudaError_t err, const char *file, int line);

// A time consuming function
__device__
double laborious_func_le_half(double x);

// Another time consuming function
__device__
double laborious_func_gt_half(double x);

// The two versions of the work. The second is expected to be faster.
// Both should receive the work at "arr" and return at "results".
void launch_gpu_work_v1(double *arr, double **results);
void launch_gpu_work_v2(double *arr, double **results);

#endif
