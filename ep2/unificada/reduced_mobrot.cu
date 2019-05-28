#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <string>
#include <omp.h>
#include <cuComplex.h>

using namespace std;

#define ITERATIONS 1000

// inline void setColorValue(png_byte *ptr, double val)
// {
// 	int v = (int)(val * 767);
// 	if (v < 0)
// 		v = 0;
// 	if (v > 767)
// 		v = 767;
// 	int offset = v % 256;

// 	if (v < 256)
// 	{
// 		ptr[0] = 0;
// 		ptr[1] = 0;
// 		ptr[2] = offset;
// 	}
// 	else if (v < 512)
// 	{
// 		ptr[0] = 0;
// 		ptr[1] = offset;
// 		ptr[2] = 255 - offset;
// 	}
// 	else
// 	{
// 		ptr[0] = offset;
// 		ptr[1] = 255 - offset;
// 		ptr[2] = 0;
// 	}
// }











__global__ void mbrot_func_gpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions, float *buffer_image)
{
	//r is for real, i for imaginary

	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;
	int max_t = 0;

	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < w * h; i += stride)
	{
		int y = i / w;
		int x = i % w;
		cuDoubleComplex current = make_cuDoubleComplex(0, 0);
		cuDoubleComplex last = make_cuDoubleComplex(0, 0);
		cuDoubleComplex c = make_cuDoubleComplex((double)c0_r + (x * d_x), (double) c0_i + (y * d_y));
		double abs = 0.0;
		bool mandel = 1;

		for (int t = 1; t < iteractions; ++t)
		{
			current = cuCadd(cuCmul(last, last), c);
			abs = cuCabs(current);
			if (abs > 2)
			{
				mandel = 0;
				if (t > max_t)
				{
					max_t = t;
				}
				buffer_image[y * w + x] = (float)t;
				break; // pintar baseado no t em que parou
			}
			last = current;
		}
		if (mandel)
		{
			buffer_image[y * w + x] = 0.0;
		}
	}

	// for (int y = 0; y < h; ++y) {
	// 	for (int x = 0; x < w; ++x) {
	// 		buffer_image[y*w + x]=buffer_image[y*w + x]/ (float) max_t;
	// 	}
	// }
}

int main(int argc, char *argv[])
{
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}

	double C0_REAL = double(atof(argv[1]));
	double C0_IMAG = double(atof(argv[2]));
	double C1_REAL = double(atof(argv[3]));
	double C1_IMAG = double(atof(argv[4]));
	int WIDTH = atoi(argv[5]);
	int HEIGHT = atoi(argv[6]);
	string CPU_GPU = argv[7];
	int THREADS = atoi(argv[8]);
	string SAIDA = argv[9];

// int max_threads;
	if (CPU_GPU == "CPU")
	{
		return 0;
	}
	else
	{
		int blockSize = 256;
		int numBlocks = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
		float *buffer_image;
		cudaMallocManaged(&buffer_image, WIDTH * HEIGHT * sizeof(float));
		if (buffer_image == NULL)
		{
			cerr << "Falha ao criar o Buffer da imagem." << endl;
			return -1;
		}
		mbrot_func_gpu<<<numBlocks, blockSize>>>(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS, buffer_image);
		cout << buffer_image[20] << endl;
		cudaDeviceSynchronize();
		cudaFree(buffer_image);
		return 0;
		// return printImage_gpu(SAIDA, WIDTH, HEIGHT, buffer_image);
	}

} //double* buffer=mbrot_func( 0.404583165379,0.234141469049,0.404612286758,0.234170590428, 1000,1000,1000);
