#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <png.h>
#include <string>
#include <omp.h>
#include <cuComplex.h>

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

int printImage_cpu(string file_name, int w, int h, double *buffer_image)
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

double *mbrot_func_cpu(double c0_r, double c0_i, double c1_r, double c1_i, int w, int h, int iteractions)
{
	//r is for real, i for imaginary

	double *buffer_image = (double *)malloc(w * h * sizeof(double));
	if (buffer_image == NULL)
	{
		cerr << "Falha ao criar o Buffer da imagem." << endl;
		return NULL;
	}

	double d_x = (c1_r - c0_r) / (double)w;
	double d_y = (c1_i - c0_i) / (double)h;
	int max_t = 0;

#pragma omp parallel for
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			complex<double> current = 0;
			complex<double> last = 0;
			complex<double> c = 0;
			bool mandel = 1;

			mandel = 1;
			c.real(c0_r + (x * d_x));
			c.imag(c0_i + (y * d_y));
			//cout << "c"<< c << endl;
			last = 0;
			for (int t = 1; t < iteractions; ++t)
			{
				current = last * last + c;
				if (abs(current) > 2)
				{
					mandel = 0;
					if (t > max_t)
					{
						max_t = t;
					}
					buffer_image[y * w + x] = (double)t;
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

#pragma omp parallel for
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			buffer_image[y * w + x] = buffer_image[y * w + x] / (double)max_t;
		}
	}

	return buffer_image;
}

int printImage_gpu(string file_name, int w, int h, float *buffer_image)
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

	int max_threads;
	if (CPU_GPU == "CPU")
	{
		max_threads = omp_get_max_threads();
		if (THREADS > max_threads)
		{
			clog << "*Warning:Nº de Threads pedido maior que o máximo aparentemente suportado.*" << endl;
		}
		omp_set_num_threads(THREADS);
		double *buffer = mbrot_func_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS);
		return printImage_cpu(SAIDA, WIDTH, HEIGHT, buffer);
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
		return printImage_gpu(SAIDA, WIDTH, HEIGHT, buffer_image);
	}

} //double* buffer=mbrot_func( 0.404583165379,0.234141469049,0.404612286758,0.234170590428, 1000,1000,1000);
