#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <png.h>
#include "mpi.h"
using namespace std;

float *main_cpu(float C0_REAL, float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, string CPU_GPU, int THREADS, string SAIDA);
int printImage(string file_name, int w, int h, float *buffer_image);
float maximize(float *array, int array_size);
void normalizeBuffer_cpu(float *buffer_image, int buffer_size, float buffer_max);

int main(int argc, char *argv[])
{
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}

	int numtasks, rank, len, rc;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	// Inicializa MPI, pega número de tasks, rank e nome do processo
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(hostname, &len);
	//cout <<"Number of tasks= " << numtasks << " My rank= " << rank <<  " Running on " << hostname << endl;

	// int result=0;

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

	if(numtasks>HEIGHT)
	{
		numtasks=HEIGHT;
	}

	if(numtasks==1){
		float *buffer_image = main_cpu(C0_REAL, C0_IMAG, C1_REAL, C1_IMAG, WIDTH,HEIGHT, CPU_GPU, THREADS, SAIDA);
		normalizeBuffer_cpu(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		MPI_Finalize();
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);
	}
	else{
		if (rank == numtasks - 1)
		{
		// Instanciam-se variáveis relativas aos parâmetros de entrada:
		int chunck = HEIGHT / (numtasks - 1);
		int chunck_resto = HEIGHT % (numtasks - 1);
		float *result;
		if (chunck_resto != 0)
		{
			float DeltaY = (C1_IMAG - C0_IMAG) / HEIGHT;
			float C0_IMAG_resto = C0_IMAG + rank * chunck * DeltaY;
			float C1_IMAG_resto = C0_IMAG_resto + (chunck_resto + 1) * DeltaY;
			//cout << rank << ": " << C0_IMAG_resto << " -> " << C1_IMAG_resto << endl;
			result = main_cpu(C0_REAL, C0_IMAG_resto, C1_REAL, C1_IMAG_resto, WIDTH, chunck + 1, CPU_GPU, THREADS, SAIDA);
			// cout << "master result " << result << endl;}
			// 	for(int j=0; j<chunck_resto*WIDTH;j++){
			// cout << result[j] << " ";}
			// cout << endl;
		}

		float *result_filho = (float *)malloc(chunck * WIDTH * sizeof(float));
		float *buffer_image = (float *)malloc(HEIGHT * WIDTH * sizeof(float));
		for (int i = 0; i < numtasks - 1; i++)
		{
			MPI_Recv(&buffer_image[i * chunck * WIDTH], WIDTH * chunck, MPI_FLOAT, i, MPI_ANY_TAG, MPI_COMM_WORLD, &Stat);
			// cout << i << ": ";
			// for(int j=0; j<chunck*WIDTH;j++){
			// cout << result_filho[j] << " ";}
			// cout << endl;
		}
		if (chunck_resto != 0)
		{
			for (int i = 0; i < chunck_resto * WIDTH; i++)
			{
				buffer_image[(HEIGHT - chunck_resto) * WIDTH + i] = result[i];
			}
		}
		// for(int j=0; j<WIDTH*HEIGHT;j++){
		// // cout << buffer_image[j] << " ";}
		// // cout << endl;
		normalizeBuffer_cpu(buffer_image, WIDTH * HEIGHT, maximize(buffer_image, WIDTH * HEIGHT));
		// Gera a imagem e retorna:
		MPI_Finalize();
		return printImage(SAIDA, WIDTH, HEIGHT, buffer_image);

		// result= main_cpu(C0_REAL,C0_IMAG, C1_REAL,C1_IMAG,WIDTH,HEIGHT,CPU_GPU,THREADS,SAIDA);
	}
	else
	{
		if(rank<numtasks){
		int chunck = HEIGHT / (numtasks - 1);
		float DeltaY = (C1_IMAG - C0_IMAG) / HEIGHT;
		float C0_IMAG_processo = C0_IMAG + rank * chunck * DeltaY;
		float C1_IMAG_processo = C0_IMAG_processo + (chunck - 1) * DeltaY;
		//cout << rank << ": " << C0_IMAG_processo << " -> " << C1_IMAG_processo << endl;
		float *result = main_cpu(C0_REAL, C0_IMAG_processo, C1_REAL, C1_IMAG_processo, WIDTH, chunck, CPU_GPU, THREADS, SAIDA);
		// cout << rank << ": " << endl;
		// for(int j=0; j<chunck*WIDTH;j++){
		// cout << result[j] << " ";}
		// cout << endl;

		MPI_Send(result, WIDTH * chunck, MPI_FLOAT, numtasks - 1, 0, MPI_COMM_WORLD);
	}}
	MPI_Finalize();
	return 0;
}
}