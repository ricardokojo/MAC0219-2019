#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h>
#include <png.h>
#include <thrust/complex.h>
using namespace std;

#define ITERATIONS 1000

inline void setColorValue(png_byte *ptr, double val)
{
	int v = (int)(val * 767);
	if (v < 0)
		v = 0;
	if (v > 767)
		v = 767;
	int offset = v % 256;

	if (v < 256)
	{
		ptr[0] = 0;
		ptr[1] = 0;
		ptr[2] = offset;
	}
	else if (v < 512)
	{
		ptr[0] = 0;
		ptr[1] = offset;
		ptr[2] = 255 - offset;
	}
	else
	{
		ptr[0] = offset;
		ptr[1] = 255 - offset;
		ptr[2] = 0;
	}
}

int printImage(string file_name, int w, int h, float *buffer_image)
{

	FILE *file = NULL;
	png_structp image_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep buffer_row = NULL;

	int status = 1;
	file = fopen(file_name.c_str(), "wb");
	if (file == NULL)
	{
		cerr << "Falha arquivo destinado para a escrita da imagem: " << file_name << endl;
		status = 0;
	}

	if (status)
	{
		image_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (image_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de escrita." << endl;
			status = 0;
		}
	}

	if (status)
	{
		info_ptr = png_create_info_struct(image_ptr);
		if (info_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de meta-dados para a imagem." << endl;
			status = 0;
		}
	}

	if (status)
	{
		if (setjmp(png_jmpbuf(image_ptr)))
		{
			cerr << "Erro durante a criação da imagem." << endl;
			status = 0;
		}
	}

	if (status)
	{
		png_init_io(image_ptr, file);
		png_set_IHDR(image_ptr, info_ptr, w, h,
								 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
								 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
		png_write_info(image_ptr, info_ptr);

		buffer_row = (png_bytep)malloc(3 * w * sizeof(png_byte));
		int y, x;
		for (y = 0; y < h; y++)
		{
			for (x = 0; x < w; x++)
			{
				setColorValue(&(buffer_row[x * 3]), buffer_image[y * w + x]);
			}
			png_write_row(image_ptr, buffer_row);
		}
		png_write_end(image_ptr, NULL);
	}

	if (file != NULL)
		fclose(file);
	if (info_ptr != NULL)
		png_free_data(image_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (image_ptr != NULL)
		png_destroy_write_struct(&image_ptr, (png_infopp)NULL);
	if (buffer_row != NULL)
		free(buffer_row);

	return status - 1;
}

float *mbrot_func_cpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions)
{
	//r is for real, i for imaginary

	float *buffer_image = (float *)malloc(w * h * sizeof(float));
	if (buffer_image == NULL)
	{
		cerr << "Falha ao criar o Buffer da imagem." << endl;
		return NULL;
	}

	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;

#pragma omp parallel for
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			thrust::complex<float> current;
			current.real(0);
			current.imag(0);
			thrust::complex<float> last;
			last.real(0);
			last.imag(0);
			thrust::complex<float> c;
			c.real(c0_r + (x * d_x));
			c.imag(c0_i + (y * d_y));
			bool mandel = 1;
			for (int t = 1; t < iteractions; ++t)
			{
				current = last * last + c;
				if (thrust::abs(current) > 2)
				{
					mandel = 0;
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
	}

	// #pragma omp parallel for
	// 	for (int y = 0; y < h; ++y)
	// 	{
	// 		for (int x = 0; x < w; ++x)
	// 		{
	// 			buffer_image[y * w + x] = buffer_image[y * w + x] / (double)max_t;
	// 		}
	// 	}

	return buffer_image;
}

__global__ void mbrot_func_gpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions, float *buffer_image)
{
	//r is for real, i for imaginary

	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("threadIDX: %d\t",index);
	//printf("blockDim: %d\t",stride);

	for (int i = index; i < w * h; i += stride)
	{
		int y = i / w;
		int x = i % w;
		
		thrust::complex<float> current;
		current.real(0);
		current.imag(0);
		
		thrust::complex<float> last;
		last.real(0);
		last.imag(0);
		
		thrust::complex<float> c;
		c.real(c0_r + (x * d_x));
		c.imag(c0_i + (y * d_y));
		
		//printf("%d ",i);
		float abs = 0.0;
		bool mandel = 1;

		for (int t = 1; t < iteractions; ++t)
		{
			current = last * last + c;
			abs = thrust::abs(current);
			if (abs > 2)
			{
				mandel = 0;
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
}

float maximize(float *array, int array_size)
{
	float max = 757.0;

	for (int i = 0; i < array_size; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
			// printf("hmmm: %d -> %f",i, max);
		}
	}
	return max;
}

void normalizeBuffer_cpu(float *buffer_image, int buffer_size, float buffer_max)
{
#pragma omp parallel for
	for (int i = 0; i < buffer_size; i++)
	{
		buffer_image[i] = buffer_image[i] / buffer_max;
	}
}

__global__ void normalizeBuffer_gpu(float *buffer_image, int buffer_size, float buffer_max)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < buffer_size; i += stride)
	{
		buffer_image[i] = buffer_image[i] / buffer_max;
	}
}

int main(int argc, char *argv[])
{
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}

	float C0_REAL = float(atof(argv[1]));
	float C0_IMAG = float(atof(argv[2]));
	float C1_REAL = float(atof(argv[3]));
	float C1_IMAG = float(atof(argv[4]));
	int WIDTH = atoi(argv[5]);
	int HEIGHT = atoi(argv[6]);
	string CPU_GPU = argv[7];
	int THREADS = atoi(argv[8]);
	string SAIDA = argv[9];

	int max_threads;
	if (CPU_GPU == "CPU")
	{
		max_threads = omp_get_max_threads();
		if (THREADS > max_threads)
		{
			clog << "*Warning:Nº de Threads pedido maior que o máximo aparentemente suportado.*" << endl;
		}
		
		omp_set_num_threads(THREADS);
		float *buffer_image = mbrot_func_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS);
		
		normalizeBuffer_cpu(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
	}
	else
	{
		int blockSize = THREADS;
		int numBlocks = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
		float *buffer_image;
		
		cudaMallocManaged(&buffer_image, WIDTH * HEIGHT * sizeof(float));
		if (buffer_image == NULL)
		{
			cerr << "Falha ao criar o Buffer da imagem." << endl;
			return -1;
		}
		
		mbrot_func_gpu<<<numBlocks, blockSize>>>(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS, buffer_image);
		cudaDeviceSynchronize();
		
		normalizeBuffer_gpu<<<numBlocks, blockSize>>>(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		cudaDeviceSynchronize();
		
		int result = printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
		cudaFree(buffer_image);
		
		return result;
	}
}
