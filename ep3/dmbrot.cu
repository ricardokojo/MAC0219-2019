//*******************************************//
//MAC5742 EP2                                //
//EP2 - Mandelbrot                           //
//Bruna Bazaluk, Felipe Serras Ricardo Kojo  //
//*******************************************//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h> //OpneMP
#include <png.h> //Manipulação de pngs
#include <thrust/complex.h> //Manipulação de números complexos para CPU e GPU
#include "mpi.h"
using namespace std;

#define ITERATIONS 1000 //Número máximo de iterações no cálculo da pertencência ao conjunto de Madelbrot


//Função que transforma um vetor de valores entre 0 a 1 em cores na escala RGP
//Baseado na função apresentada em 	http:;;www.labbookpages.co.uk/software/imgProc/libPNG.html
inline void setColorValue(png_byte *ptr, double val)
{
	int v = (int)(val * 767);
	if (v < 0)
		v = 0;
	if (v > 767)
		v = 767;
	int offset = v % 256;

	if (v < 100)
	{
		ptr[0] = 0;
		ptr[1] = 0;
		ptr[2] = offset;
	}
	else if (v < 300)
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


//Função que recebe uma um vetor de floats representado a imagem de buffer e o salva no arquivo png 
// com o nome filname
int printImage(string file_name, int w, int h, float *buffer_image)
{
	//A cada oassi atualiza-se o inteiro status que controla se o fluxo de salvamento da imagem deve prosseguir.
	//Se houver algum erro num passo intermediário o fluxo é interrompido.

	FILE *file = NULL;
	png_structp image_ptr = NULL;
	png_infop info_ptr = NULL;
	png_bytep buffer_row = NULL;

	//Criação do Arquivo:
	int status = 1;
	file = fopen(file_name.c_str(), "wb");
	if (file == NULL)
	{
		cerr << "Falha arquivo destinado para a escrita da imagem: " << file_name << endl;
		status = 0;
	}

	//Alocação da estrutura de escrita:
	if (status)
	{
		image_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
		if (image_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de escrita." << endl;
			status = 0;
		}
	}

	//Alocação da estrutura de meta-dados (informações complementares do arquivo):
	if (status)
	{
		info_ptr = png_create_info_struct(image_ptr);
		if (info_ptr == NULL)
		{
			cerr << "Falha ao alocar uma estrutura de meta-dados para a imagem." << endl;
			status = 0;
		}
	}
	//Criação da imagem:
	if (status)
	{
		if (setjmp(png_jmpbuf(image_ptr)))
		{
			cerr << "Erro durante a criação da imagem." << endl;
			status = 0;
		}
	}

	// Povoamento do arquivo, tanto com as informações complementares quanto com as cores:
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
			png_write_row(image_ptr, buffer_row); //A trasferência do buffer para a imagem é feita linha por linha.
		}
		png_write_end(image_ptr, NULL);
	}

	//Em qualquer caso todas as estruturas criadas são fechadas e finalizadas:
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


//Função de geração da imagem de buffer para cpu:
float *mbrot_func_cpu(float c0_r, float c0_i, float c1_r, float c1_i, int w, int h, int iteractions)
{
	//Aloca-se a imagem de buffer::
	float *buffer_image = (float *)malloc(w * h * sizeof(float));
	if (buffer_image == NULL)
	{
		cerr << "Falha ao criar o Buffer da imagem." << endl;
		return NULL;
	}

	float d_x = (c1_r - c0_r) / (float)w;
	float d_y = (c1_i - c0_i) / (float)h;

//Para cada pixel da imagem realiza-se o teste de pertencência:
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
			thrust::complex<float> c ;
			//Gera-se o valor complexo correspondente ao pixel
			c.real(c0_r + (x * d_x));
			c.imag(c0_i + (y * d_y));
			bool mandel = 1;
			//Aplica-se a função de Madelbrot sobre o valor até o número de iterações pré-definido:
			for (int t = 1; t < iteractions; ++t)
			{
				current = last * last + c;
				//Caso o valor atual tenha passado de 2, o valor não pertence ao conjunto
				//e o valor do pixel correspondente é o numero de iterações necessário para perceber
				//a não-pertencência:
				if (thrust::abs(current) > 2)
				{
					mandel = 0;
					buffer_image[y * w + x] = (float)t;
					break; // pintar baseado no t em que parou
				}
				last = current;
			}
			//Caso contrário, o valor pertence ao conjunto e recebe valor 0:
			if (mandel)
			{
				buffer_image[y * w + x] = 0.0;
			}
		}
	}

	return buffer_image;
}

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

//Função auxiliar que encontra o valor máximo num array de floats.
//(Utilizada para a normaização do buffer antes da geração da imagem)
float maximize(float* array, int array_size){
	float max=757.0;

	for(int i=0; i<array_size;i++){
		if(array[i]>max){
			max=array[i];
		}
	}
return max;
}

//Versão de normalização de buffer para a cpu:
void normalizeBuffer_cpu(float *buffer_image, int buffer_size, float buffer_max){
	#pragma omp parallel for
	for(int i=0; i<buffer_size; i++){
		buffer_image[i]=buffer_image[i]/buffer_max;
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



//Função principal:
int main(int argc, char *argv[])
{
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}
	//Instanciam-se variaǘeis relativas aos parâmetros de entrada:
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

	int  numtasks, rank, len, rc; 
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
	MPI_Get_processor_name(hostname, &len);
	cout <<"Number of tasks= " << numtasks << " My rank= " << rank <<  " Running on " << hostname << endl;
	if(rank==0){

	
	//Verifica-se se é o caso de execução com GPU ou CPU:
	if (CPU_GPU == "CPU")
	{
		//Caso seja CPU verifica-se se o número de threads pedido está acima do aceito pela cpu:
		max_threads = omp_get_max_threads();
		if (THREADS > max_threads)
		{
			clog << "*Warning:Nº de Threads pedido maior que o máximo aparentemente suportado.*" << endl;
		}
		omp_set_num_threads(THREADS); //Define-se o numero de threads a ser utilizado pelo openmp

		//Produz-se o buffer:
		float *buffer_image = mbrot_func_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH, HEIGHT, ITERATIONS);
		//Normaliza-se a imagem:
		normalizeBuffer_cpu(buffer_image,WIDTH*HEIGHT,maximize(buffer_image,WIDTH*HEIGHT));
		//Gera a imagem e retorna:
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
	}
	else
	{
		//Innstancia-se o número de threads e o número de blocos:
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
}
else{
	return 0;
}

} 
