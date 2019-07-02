#ifndef _BOOTSTRAP_H
#define _BOOTSTRAP_H


// Performs some useless memcpy and kernel launchs to "warmup" the GPU.
// This was necessary as some people reported major delays in the first kernel
// launches for some GPUs, probably due to initialization.
void warm_gpu_up(void);

// Tunes the test parameter GPU_WORK_ITERATIONS, which is the number of
// iterations performed by the main loop at the gpu_work functions.
// This is done in order to ensure no faster (or slower) GPU gets
// penalized during tests.
int tune_iterations(double *arr);

#endif
