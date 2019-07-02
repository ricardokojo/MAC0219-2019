#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include "utils.h"
#include "bootstrap.h"

#define DTOL 1e-9
#define MIN_SPEEDUP 1.4
#define LABORIOUS_ITERATIONS 100

__device__ int GPU_WORK_ITERATIONS;

void cudaCheckError(cudaError_t err, const char *file, int line)
{
        if (err != cudaSuccess)
                DIE("CUDA error at %s:%d: %s\n", file, line,
                    cudaGetErrorString(err));
}

// A time consuming function
__device__
double laborious_func_le_half(double x)
{
        for(int i = 0; i < LABORIOUS_ITERATIONS; ++i) {
                if (i % 2 == 0)
                        x -= 0.0012;
                else
                        x += 0.0021;
        }
        return x;
}

// Another time consuming function
__device__
double laborious_func_gt_half(double x)
{
        for(int i = 0; i < LABORIOUS_ITERATIONS; ++i) {
                if (i % 2 == 0)
                        x += 0.0012;
                else
                        x -= 0.0021;
        }
        return x;
}

int check_results(double *reference, double *result)
{
        for (int i = 0; i < ARR_SIZE; ++i) {
                if (abs(reference[i] - result[i]) > DTOL)
                        return 0;
        }
        return 1;
}

void randomly_fill_array(double *v, int size)
{
        srand(1373); // arbitrary initialization
        for (int i = 0; i < size; ++i)
                v[i] = (double)rand() / RAND_MAX;
}

int main(int argc, char **argv)
{
        static double arr[ARR_SIZE];
        struct timeval start, end;
        double *results_v1, *results_v2;
        double elapsed_v1, elapsed_v2, speedup;
        int correctness, gpu_work_iterations;
        int ret = 0;

        results_v1 = (double *) malloc(ARR_SIZE * sizeof(double));
        results_v2 = (double *) malloc(ARR_SIZE * sizeof(double));
        if (!results_v1 || !results_v2)
                DIE("failed to malloc at main\n");

        randomly_fill_array(arr, ARR_SIZE);
        printf("Warming up...               ");
        warm_gpu_up();
        printf("done.\n");

        printf("Tunning test parameters...  ");
        fflush(stdout);
        gpu_work_iterations = tune_iterations(arr);
        if (gpu_work_iterations <= 0)
                DIE("Failed to tune... Too fast GPU?\n");
        cudaAssert(cudaMemcpyToSymbol(GPU_WORK_ITERATIONS, &gpu_work_iterations,
                                      sizeof(int)));
        printf("done.\n");
        printf("Note: using GPU_WORK_ITERATIONS = %d\n", gpu_work_iterations);
        printf("(based on tunning for this machine)\n\n");

        printf("Running v1...               ");
        fflush(stdout);
        gettimeofday(&start, NULL);
        launch_gpu_work_v1(arr, &results_v1);
        gettimeofday(&end, NULL);
        elapsed_v1 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("done.\n");

        printf("Running v2...               ");
        fflush(stdout);
        gettimeofday(&start, NULL);
        launch_gpu_work_v2(arr, &results_v2);
        gettimeofday(&end, NULL);
        elapsed_v2 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("done.\n\n");

        // Check results and time
        printf("Results\n===================\n");
        correctness = check_results(results_v1, results_v2);
        speedup = elapsed_v1 / elapsed_v2;
        printf("v1: %.4fs\n", elapsed_v1);
        printf("v2: %.4fs\n", elapsed_v2);
        printf("speedup: %.4fx\n\n", speedup);

        if (correctness) {
                printf("Correctness: OK\n");
        } else {
                ret = 1;
                printf("Correctness: FAILED (wrong results)\n");
        }

        if (speedup >= MIN_SPEEDUP) {
                printf("Speedup: OK\n");
        } else {
                ret = 1;
                printf("Speedup: FAILED (min: %.2f)\n", MIN_SPEEDUP);
        }

        free(results_v1);
        free(results_v2);
        return ret;
}
