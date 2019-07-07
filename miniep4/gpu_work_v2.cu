#include "utils.h"

#define THS_PER_BLOCK 256

__global__ void gpu_work_v2(double *arr)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id >= ARR_SIZE)
        return;

    int delayed = 0;
    int cond;
    double a = arr[id];

    for (int i = 0; i < GPU_WORK_ITERATIONS;)
    {
        if (!delayed)
            cond = (a <= 0.5);

        int cond_for_all = !cond_for_all;
        delayed = (cond_for_all != cond);
        if (!delayed)
        {
            if (cond_for_all)
            {
                a = laborious_func_le_half(a);
            }
            else
            {
                a = laborious_func_gt_half(a);
            }
            ++i;
        }
    }

    arr[id] = a;
}

// Launch the work on arr and return it at results;
void launch_gpu_work_v2(double *arr, double **results)
{
    double *d_arr;

    cudaAssert(cudaMalloc(&d_arr, ARR_SIZE * sizeof(double)));
    cudaAssert(cudaMemcpy(d_arr, arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyHostToDevice));

    gpu_work_v2<<<DIV_CEIL_INT(ARR_SIZE, THS_PER_BLOCK), THS_PER_BLOCK>>>(d_arr);
    cudaAssert(cudaDeviceSynchronize());

    cudaAssert(cudaMemcpy(*results, d_arr, ARR_SIZE * sizeof(double),
                          cudaMemcpyDeviceToHost));
    cudaAssert(cudaFree(d_arr));
}
