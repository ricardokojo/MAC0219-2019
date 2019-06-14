#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <png.h>
#include "mpi.h"
using namespace std;

int main_cpu(float C0_REAL,float C0_IMAG, float C1_REAL, float C1_IMAG, int WIDTH, int HEIGHT, string CPU_GPU, int THREADS, string SAIDA);

int main(int argc, char *argv[])
{
	if (argc < 9)
	{
		cout << " uso: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <SAIDA>\n";
		return 1;
	}

	int  numtasks, rank, len, rc; 
char hostname[MPI_MAX_PROCESSOR_NAME];

// initialize MPI  
MPI_Init(&argc,&argv);

// get number of tasks 
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

// get my rank  
MPI_Comm_rank(MPI_COMM_WORLD,&rank);

// this one is obvious  
MPI_Get_processor_name(hostname, &len);
cout <<"Number of tasks= " << numtasks << " My rank= " << rank <<  " Running on " << hostname << endl;

int result=0;

// done with MPI  

	if(rank==0){
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

	result= main_cpu(C0_REAL,C0_IMAG, C1_REAL,C1_IMAG,WIDTH,HEIGHT,CPU_GPU,THREADS,SAIDA);
}
else{
	result= 0;

}
MPI_Finalize();
return result;
}
