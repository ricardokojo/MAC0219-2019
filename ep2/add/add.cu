#include <stdio.h>
#include <stdlib.h>

#define DIE(...) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
}

#define ABS(x) ((x) < 0 ? (-x) : (x))

void cpu_add(REAL *a, REAL *b, REAL *c, unsigned n) {
        for (int i = 0; i < n; ++i)
                c[i] = a[i] + b[i];
}

void cudaAssert(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Erro!\n");
        abort();
    }
}

__global__
void gpu_add_kernel(REAL *a, REAL *b, REAL *c, unsigned n)
{
    const int globalIndex = blockDim.x*blockIdx.x + threadIdx.x;

    if (globalIndex < n)
        c[globalIndex] = a[globalIndex] + b[globalIndex];
}

void gpu_add(REAL *a, REAL *b, REAL *c, unsigned n) {
        const int THREADS_PER_BLOCK = 128;
        const int NUM_BLOCKS = (n + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;

        REAL* d_a, *d_b, *d_c;

        cudaAssert(cudaMalloc(&d_a, n*sizeof(*d_a)));
        cudaAssert(cudaMalloc(&d_b, n*sizeof(*d_b)));
        cudaAssert(cudaMalloc(&d_c, n*sizeof(*d_c)));

        cudaAssert(cudaMemcpy(d_a, a, n*sizeof(*a), cudaMemcpyHostToDevice));
        cudaAssert(cudaMemcpy(d_b, b, n*sizeof(*b), cudaMemcpyHostToDevice));

        gpu_add_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
        cudaAssert(cudaDeviceSynchronize());

        cudaAssert(cudaMemcpy(c, d_c, n*sizeof(*c), cudaMemcpyDeviceToHost));

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
}

int check_add(REAL *a, REAL *b, REAL *c, unsigned n) {
        for (int i = 0; i < n; ++i) {
                REAL sum = a[i] + b[i];
                if (ABS(sum - c[i]) > 0.0001)
                        return 0;
        }
        return 1;
}

void random_fill(REAL *a, unsigned n) {
        for (int i = 0; i < n; ++i)
                a[i] = (REAL) rand() / RAND_MAX;
}

int main(int argc, char **argv) {
        unsigned n;
        REAL *a, *b, *c;

        if (argc != 2 || sscanf(argv[1], "%u", &n) != 1)
                DIE("use: %s <array_size>\n", argv[0]);

        srand(1337);

        if ((a = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");
        if ((b = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");
        if ((c = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");

        random_fill(a, n);
        random_fill(b, n);

        gpu_add(a, b, c, n);

        if(!check_add(a, b, c, n))
                printf("sum FAILED\n");
        else
                printf("sum correct\n");

        return 0;
}
