#include "utils.h"
#include <assert.h>

#define THS_PER_BLOCK 256
#define NUM_BLOCKS 40

__global__
void gpu_work_v2_le(double *arr)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < WORK_ITERATIONS_LE; ++i)
        arr[id] = next_step_le_half(arr[id]);
}

__global__
void gpu_work_v2_gt(double *arr)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < WORK_ITERATIONS_GT; ++i)
        arr[id] = next_step_gt_half(arr[id]);
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
    // Launch the GPU kernel
    double *d_arr, arr_le[ARR_SIZE], arr_gt[ARR_SIZE];
    assert(ARR_SIZE == THS_PER_BLOCK * NUM_BLOCKS);

    // copies elements from arr to arr_aux if lt 0.5
    // else -1
    (arr, &arr_le, &arr_gt)
    // send to GPU calculate
    // copy from GPU array to result array

    cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
    cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
        cudaMemcpyHostToDevice));

    gpu_work_v2_le<<<NUM_BLOCKS, THS_PER_BLOCK>>>(d_arr);
    cudaAssert(cudaDeviceSynchronize());

    gpu_work_v2_gt<<<NUM_BLOCKS, THS_PER_BLOCK>>>(d_arr);
    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
        cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_arr));
}

void check_lt(double *arr, double *arr_aux) {
    for (int i = 0; i < ARR_SIZE; ++i) {
        if (arr[i] <= 0.5) {
            arr_aux = arr[i];
        } else {
            arr_aux = -1;
        }
    }
}

