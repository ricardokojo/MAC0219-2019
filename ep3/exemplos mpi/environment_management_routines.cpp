// required MPI include file  
#include "mpi.h"
#include <stdio.h>

using namespace std;

int main(int argc, char *argv[]) {
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

// done with MPI  
MPI_Finalize();
}