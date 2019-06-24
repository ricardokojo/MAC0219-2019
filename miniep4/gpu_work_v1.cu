#include "utils.h"
#include <assert.h>


#define THS_PER_BLOCK 256
#define NUM_BLOCKS 40

__global__
void gpu_work_v1(double *arr)
{
        const int id = blockDim.x * blockIdx.x + threadIdx.x;

        if (arr[id] <= 0.5) {
                for (int i = 0; i < WORK_ITERATIONS_LE; ++i)
                        arr[id] = next_step_le_half(arr[id]);
        } else {
                for (int i = 0; i < WORK_ITERATIONS_GT; ++i)
                        arr[id] = next_step_gt_half(arr[id]);
        }
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v1(double *arr, double **results)
{
        double *d_arr;
        assert(ARR_SIZE == THS_PER_BLOCK * NUM_BLOCKS);

        cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
        cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyHostToDevice));

        gpu_work_v1<<<NUM_BLOCKS, THS_PER_BLOCK>>>(d_arr);
        cudaAssert(cudaDeviceSynchronize());

        cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                              cudaMemcpyDeviceToHost));
        cudaAssert(cudaFree(d_arr));
}
