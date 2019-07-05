#include "utils.h"
#include <assert.h>

#define THS_PER_BLOCK 256
#define UNTIL_NORM 100

__global__
void gpu_work_le(double *arr, bool *is_less)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= ARR_SIZE || !is_less[id]) return;

    for(int i = 0; i < (GPU_WORK_ITERATIONS-UNTIL_NORM); ++i) {
        if (arr[id] <= 0.5)
            arr[id] = laborious_func_le_half(arr[id]);
        else
            arr[id] = laborious_func_gt_half(arr[id]);
    }
}

__global__
void gpu_work_gt(double *arr, bool *is_less)
{
    const int real_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int id = real_id % ARR_SIZE;

    if (real_id >= ARR_SIZE && is_less[id]) return;
    if (real_id < ARR_SIZE && !is_less[id]) return;

    for(int i = UNTIL_NORM; i < GPU_WORK_ITERATIONS; ++i) {
        if (arr[id] <= 0.5)
            arr[id] = laborious_func_le_half(arr[id]);
        else
            arr[id] = laborious_func_gt_half(arr[id]);
    }
}

__global__
void gpu_work_stable(double *arr, bool *is_less)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= ARR_SIZE) return;

    for(int i = 0; i < GPU_WORK_ITERATIONS && i < UNTIL_NORM; ++i) {
        if (arr[id] <= 0.5)
            arr[id] = laborious_func_le_half(arr[id]);
        else
            arr[id] = laborious_func_gt_half(arr[id]);
    }

    is_less[id] = (arr[id] <= 0.5);
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
    double *d_arr;
    bool *d_is_less;

    cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
    cudaAssert(cudaMalloc(&d_is_less, ARR_SIZE * sizeof(bool)));

    cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyHostToDevice));

    gpu_work_stable<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr, d_is_less);
    cudaAssert(cudaDeviceSynchronize());

    // gpu_work_le<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr, d_is_less);
    gpu_work_gt<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK)*2, THS_PER_BLOCK>>>(d_arr, d_is_less);
    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaAssert(cudaFree(d_arr));
    cudaAssert(cudaFree(d_is_less));
}