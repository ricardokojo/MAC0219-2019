//********************************************//
// MAC0219/5742 - EP3                         //
// EP3 - Mandelbrot                           //
// Bruna Bazaluk, Felipe Serras, Ricardo Kojo //
//********************************************//
//*Arquivo peiincipal que chama as funções que envolvem MPI e fazem a administração e divisão do trabalho.
// Essas divisão e feita com base nas linhas. Bocos de linhas, ou seja, blocos com a mesma largura da imagem são
//distriuídos entre os diferentes processos. O processo mestre trata das linhas restantes da imagem e concatena 
//todos os resultados para produzir a imagem final. Ele normaliza ela e a salva no arquivo de nome pedido, usando
// as funções d processamento de imagens de img_util.cpp *//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <png.h>
#include "mpi.h"
using namespace std;

//Estabelece os Headers de arquivos externos a serem utilizados:
float *main_cpu(float C0_REAL, float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, string CPU_GPU, int THREADS, string SAIDA);
int printImage(string file_name, int w, int h, float *buffer_image);
float maximize(float *array, int array_size);
void normalizeBuffer_cpu(float *buffer_image, int buffer_size, float buffer_max);

int main(int argc, char *argv[])
{
	//Verifica superficialmente sea estrutura dos argumentos fornecida é correta:
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}

	int numtasks, rank, len, rc;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	// Inicializa MPI, pega número de tasks, rank do processo corrente e nome da máquina
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(hostname, &len);

	//Instancia os parametros de entrada
	float C0_REAL = float(atof(argv[1]));
	float C0_IMAG = float(atof(argv[2]));
	float C1_REAL = float(atof(argv[3]));
	float C1_IMAG = float(atof(argv[4]));
	int WIDTH = atoi(argv[5]);
	int HEIGHT = atoi(argv[6]);
	string CPU_GPU = argv[7];
	int THREADS = atoi(argv[8]);
	string SAIDA = argv[9];
	MPI_Status Stat;

	//Adequa o tamanho do numero de processos, caso esse seja maior do que o número de linhas da imagem:
	if(numtasks>HEIGHT)
	{
		numtasks=HEIGHT;
	}

	//Trata o caso em que há apenas um processo. Nesse caso não é necessário dividir o trabalho:
	if(numtasks==1){
		float *buffer_image = main_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH,HEIGHT, CPU_GPU, THREADS, SAIDA);
		normalizeBuffer_cpu(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		MPI_Finalize();
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
	}
	//Trata o caso em que haverá distribuição:
	else{
		//Para o Processo Mestre:
		if (rank == numtasks - 1)
		{
		// Instanciam-se variáveis relativas aos parâmetros de entrada, que ensse caso correspondem 
		// as linhas não tratadas pelos processos filhos:
		int chunck = HEIGHT / (numtasks - 1);
		int chunck_resto = HEIGHT % (numtasks - 1);
		float *result;
		
		//Processam-se as linhas restantes para o caso em que há linhas restantes a serem tratadas:
		if (chunck_resto != 0)
		{
			float DeltaY = (C1_IMAG - C0_IMAG) / HEIGHT;
			float C0_IMAG_resto = C0_IMAG + rank * chunck * DeltaY;
			float C1_IMAG_resto = C0_IMAG_resto + (chunck_resto + 1) * DeltaY;
			result = main_cpu(C0_REAL, C0_IMAG_resto, C1_REAL, C1_IMAG_resto, WIDTH, chunck + 1, CPU_GPU, THREADS, SAIDA);
		}

		//Receb-se uma a uma as mensagens dos processos filhos, contendo suas partições da matriz:
		float *buffer_image = (float *)malloc(HEIGHT * WIDTH * sizeof(float));
		for (int i = 0; i < numtasks - 1; i++)
		{
			MPI_Recv(&buffer_image[i * chunck * WIDTH], WIDTH * chunck, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &Stat);

		}

		//Concatena-se a parte da matriz processada pelo próprio pai aos resultados dos filhos:
		if (chunck_resto != 0)
		{
			for (int i = 0; i < chunck_resto * WIDTH; i++)
			{
				buffer_image[(HEIGHT - chunck_resto) * WIDTH + i] = result[i];
			}
		}
		//Normaliza-se a matriz:
		normalizeBuffer_cpu(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		//Finaliza-se o  ambiente MPI:
		MPI_Finalize();
		//Salva a imagem e retorna
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
	}
	//Para o caso dos processos filhos:
	else
	{
		// Só os filhos  até o número de linhas trabalham. Os filhos excedentes são simplesmente finalizados:
		if(rank<numtasks){
			int chunck = HEIGHT / (numtasks - 1);
			float DeltaY = (C1_IMAG - C0_IMAG) / HEIGHT;
			float C0_IMAG_processo = C0_IMAG + rank * chunck * DeltaY;
			float C1_IMAG_processo = C0_IMAG_processo + (chunck - 1) * DeltaY;
			float *result = main_cpu(C0_REAL, C0_IMAG_processo, C1_REAL, C1_IMAG_processo, WIDTH, chunck, CPU_GPU, THREADS, SAIDA);
			//Manda-se o seu resultado como mensagem para o processo Pai:
			MPI_Send(result, WIDTH * chunck, MPI_FLOAT, numtasks - 1, 0, MPI_COMM_WORLD);
	}
}
	//Finaliza o Ambiente MPI:
	MPI_Finalize();
	//Retorna:
	return 0;
}
}