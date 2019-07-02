#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include "utils.h"

#define TUNNING_ITERATIONS 100000
#define TUNNING_BIAS 9353.6
#define THS_PER_BLOCK 256

__global__
void tunner_kernel(double *arr)
{
        const int id = blockDim.x * blockIdx.x + threadIdx.x;
        double x;

        if (id >= ARR_SIZE) return;

        x = arr[id];
        for(int i = 0; i < TUNNING_ITERATIONS; ++i) {
                if (i % 2 == 0)
                        x -= 0.0012;
                else
                        x += 0.0021;
        }
        arr[id] = x;
}

// Tunes the test parameter GPU_WORK_ITERATIONS, which is the number of
// iterations performed by the main loop at the gpu_work functions.
// This is done in order to ensure no faster (or slower) GPU gets
// penalized during tests.
int tune_iterations(double *arr)
{
        double *d_arr;
        struct timeval start, end;
        double elapsed;

        cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
        cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyHostToDevice));

        gettimeofday(&start, NULL);
        tunner_kernel<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr);
        cudaAssert(cudaDeviceSynchronize());
        gettimeofday(&end, NULL);

        cudaAssert(cudaFree(d_arr));
        elapsed = (end.tv_sec - start.tv_sec) +
                  (end.tv_usec - start.tv_usec) / 1000000.0;
        return (int)(TUNNING_BIAS / elapsed);
}

__global__
void cuda_do_nothing(double *d)
{
        *d = *d + 0.0;
}

// Performs some useless memcpy and kernel launchs to "warmup" the GPU.
// This was necessary as some people reported major delays in first kernel
// launches for some GPUs, probably due to initialization.
void warm_gpu_up(void)
{
        double num = 1.0, num_back = 0.0;
        double *d_num;

        cudaAssert(cudaMalloc(&d_num, sizeof(double)));
        cudaAssert(cudaMemcpy(d_num, &num, sizeof(double),
                              cudaMemcpyHostToDevice));

        cuda_do_nothing<<<1, 1>>>(d_num);
        cudaAssert(cudaDeviceSynchronize());

        cudaAssert(cudaMemcpy(&num_back, d_num, sizeof(double),
                              cudaMemcpyDeviceToHost));
        cudaAssert(cudaFree(d_num));
        assert(num == num_back);
}
