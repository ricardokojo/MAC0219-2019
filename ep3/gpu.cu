#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thrust/complex.h> //Manipulação de números complexos para CPU e GPU
#include <png.h>
using namespace std;

#define ITERATIONS 1000

inline void setColorValue(png_byte *ptr, double val);
int printImage(string file_name, int w, int h, float *buffer_image);
float maximize(float* array, int array_size);


//Versão da função de criação da imagem buffer, que define a pertencência dos numeros complexos em relação ao conjunto de
//Mandelbrot para a gpu. Ele é em muito similar a versão da gpu. as principais diferenças encontram-se na 
//forma de percorrer os pixels:
__global__ void mbrot_func_gpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions, float *buffer_image)
{
	//Considera-se que a imagem de buffer já foi alocada, pois ela deve ser alocada na memória da gpu:
	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;

	//Para cada chamada o índice e o passo do loop são calculados em função do número da thread e o número do bloco
	//da gpu que a está executando. Isso garante que nenhuma thread realiza o mesmo trabalho que outra:
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

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

//Versão de normalização de buffer para a cpu:
__global__ void normalizeBuffer_gpu(float* buffer_image, int buffer_size, float buffer_max){
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;
		for(int i=index; i<buffer_size; i+=stride){
			buffer_image[i]=buffer_image[i]/buffer_max;
	}
}

int main_gpu(float C0_REAL, float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, int THREADS, string SAIDA){
		int blockSize = THREADS;
		int numBlocks = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
		float *buffer_image;
		cudaMallocManaged(&buffer_image, WIDTH * HEIGHT * sizeof(float)); //Aloca memória da gpu para a imagem de buffer
		if (buffer_image == NULL)
		{
			cerr << "Falha ao criar o Buffer da imagem." << endl;
			return -1;
		}
		//Gera-se a imagem de  buffer:
		mbrot_func_gpu<<<numBlocks, blockSize>>>(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS, buffer_image);
		cudaDeviceSynchronize(); //Espera-se o fim dos cálculos para continuação da parte sequenciaç
		//Normaliza o Buffer:
		normalizeBuffer_gpu<<<numBlocks,blockSize>>>(buffer_image,WIDTH*HEIGHT,maximize(buffer_image,WIDTH*HEIGHT));
		cudaDeviceSynchronize(); //Espera mais um poquinho.
		int result=printImage(SAIDA, WIDTH, HEIGHT, buffer_image); //Gera-se a imagem
		cudaFree(buffer_image); //Libera a memória do cuda alocada para o buffer
		return result; //Hora de dizer tchau.
}