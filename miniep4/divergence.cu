#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "utils.h"

#define DTOL 1e-6
#define MIN_SPEEDUP 1.5

void cudaCheckError(cudaError_t err, const char *file, int line)
{
        if (err != cudaSuccess)
                DIE("CUDA error at %s:%d: %s\n", file, line,
                    cudaGetErrorString(err));
}

// A factorial function. Implemented [almost] the worst way possible.
__device__
int factorial(int n)
{
        int ret = 1;
        for (int i = 1; i <= n; ++i)
                ret *= i;
        return ret;
}

// A step in the work for x <= 0.5
// [0, 0.5] -> [0, 0.5]
// It [almost] follows this function: https://bit.ly/2FjdsL0
__device__
double next_step_le_half(double x)
{
        double x_p = x + 0.1;
        double a = cos(-sin(cos(x_p)));
        double b = sin(x_p) * atan(sin(x_p));
        double c = sqrt(a / b) * sin((double) factorial(11)) - 0.23;
        double k = c / 3.46;
        if (k < 0) k = 0;
        if (k > 0.5) k = 0.5;
        return k;
}

// A step in the work for x > 0.5
// ]0.5, 1] -> ]0.5, 1]
// It [almost] follows this function: https://bit.ly/31D982B
__device__
double next_step_gt_half(double x)
{
        double l = sin(cos(pow(x, 3)) / pow(x, 2));
        double m = cos(atan(x + sin(x) + atan((double)factorial(9))));
        double k = ((l * m) + 0.88) / 1.2;
        if (k <= 0.5) k = 0.51;
        if (k > 1) k = 1;
        return k;
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
        double arr[ARR_SIZE];
        struct timeval start, end;
        double *results_v1, *results_v2;
        double elapsed_v1, elapsed_v2, speedup;
        int correctness;
        int ret = 0;

        results_v1 = (double *) malloc(ARR_SIZE * sizeof(double));
        results_v2 = (double *) malloc(ARR_SIZE * sizeof(double));
        if (!results_v1 || !results_v2)
                DIE("failed to malloc at main\n");

        randomly_fill_array(arr, ARR_SIZE);

        gettimeofday(&start, NULL);
        launch_gpu_work_v1(arr, &results_v1);
        gettimeofday(&end, NULL);
        elapsed_v1 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

        gettimeofday(&start, NULL);
        launch_gpu_work_v2(arr, &results_v2);
        gettimeofday(&end, NULL);
        elapsed_v2 = (end.tv_sec - start.tv_sec) +
                     (end.tv_usec - start.tv_usec) / 1000000.0;

        // Check results and time
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
