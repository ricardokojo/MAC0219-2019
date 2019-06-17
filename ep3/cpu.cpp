//********************************************//
// MAC0219/5742 - EP3                         //
// EP3 - Mandelbrot                           //
// Bruna Bazaluk, Felipe Serras, Ricardo Kojo //
//********************************************//
//*Arquivo que contem as funções para processamento em cpu.*//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h>						// OpenMP
#include <thrust/complex.h> // Manipulação de números complexos para CPU e GPU
#include <png.h>
using namespace std;

#define ITERATIONS 1000 // Número máximo de iterações no cálculo da pertencência ao conjunto de Mandelbrot

//Estabelece os Headers de arquivos externos a serem utilizados:
inline void setColorValue(png_byte *ptr, double val);
int printImage(string file_name, int w, int h, float *buffer_image);
float maximize(float *array, int array_size);
float *main_gpu(float C0_REAL, float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, int THREADS, string SAIDA);

// Função de geração da imagem de buffer para cpu:
float *mbrot_func_cpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions)
{
	// Aloca-se a imagem de buffer:
	float *buffer_image = (float *)malloc(w * h * sizeof(float));
	if (buffer_image == NULL)
	{
		cerr << "Falha ao criar o Buffer da imagem." << endl;
		return NULL;
	}

	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;

// Para cada pixel da imagem realiza-se o teste de pertencência:
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
			// Gera-se o valor complexo correspondente ao pixel
			c.real(c0_r + (x * d_x));
			c.imag(c0_i + (y * d_y));
			bool mandel = 1;
			// Aplica-se a função de Mandelbrot sobre o valor até o número de iterações pré-definido:
			for (int t = 1; t < iteractions; ++t)
			{
				current = last * last + c;
				// Caso o valor atual tenha passado de 2, o valor não pertence ao conjunto
				// e o valor do pixel correspondente é o numero de iterações necessário para perceber
				// a não-pertencência:
				if (abs(current) > 2)
				{
					mandel = 0;
					buffer_image[y * w + x] = (float)t;
					break; // pintar baseado no t em que parou
				}
				last = current;
			}
			// Caso contrário, o valor pertence ao conjunto e recebe valor 0:
			if (mandel)
			{
				buffer_image[y * w + x] = 0.0;
			}
		}
	}

	return buffer_image;
}

// Versão de normalização de buffer para a cpu:
void normalizeBuffer_cpu(float *buffer_image, int buffer_size, float buffer_max)
{
#pragma omp parallel for
	for (int i = 0; i < buffer_size; i++)
	{
		buffer_image[i] = buffer_image[i] / buffer_max;
	}
}

// Função principal para o caso CPU. Ela chama a função principal da GPU se for o caso:
float *main_cpu(float C0_REAL, float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, string CPU_GPU, int THREADS, string SAIDA)
{

	// Instancia-se variáveis relativas aos parâmetros de entrada:
	int max_threads;

	// Verifica-se se é o caso de execução com GPU ou CPU:
	if (CPU_GPU == "CPU")
	{
		// Caso seja CPU verifica-se se o número de threads pedido está acima do aceito pela cpu:
		max_threads = omp_get_max_threads();
		if (THREADS > max_threads)
		{
			cerr << "*Warning:Nº de Threads pedido maior que o máximo aparentemente suportado.*" << endl;
		}
		omp_set_num_threads(THREADS); // Define-se o numero de threads a ser utilizado pelo openmp

		// Produz-se o buffer:
		float *buffer_image = mbrot_func_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS);
		//Retorna:
		return buffer_image;
	}
	else
	{
		//Caso o processamento deva ser feito na GPU a função chama a função principal da GPU:
		return main_gpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, THREADS, SAIDA);
	}
}
