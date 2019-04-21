/* Aqui é onde os testes são implementados */

#include "matrix.h"
#include "time_extra.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Se o seu processador tiver pouco cache (muito lento), talvez seja prático
 * diminuir esse número. Use uma pot. de 2
 */
#define N 2048

int main()
{
    double *restrict A = aligned_alloc(8, N*N*sizeof(*A));
    double *restrict B = aligned_alloc(8, N*N*sizeof(*B));
    double *restrict C = aligned_alloc(8, N*N*sizeof(*C));
    double *restrict D = aligned_alloc(8, N*N*sizeof(*D));
    double *restrict res[3] = {C, D, D};

    struct timeval t[3];
	struct timeval t1, t2;
    struct timeval* previous_time = &t[0];
    int i;

    srand(1337);
    matrix_fill_rand(N, A);
    matrix_fill_rand(N, B);

    FOR_EACH_DGEMM(i)
    {
        printf("Executando dgemm_%d...\n", i);
        memset(res[i], 0, N*N*sizeof(*res[i]));
        gettimeofday(&t1, NULL);
        matrix_which_dgemm(i, N, res[i], A, B);
        gettimeofday(&t2, NULL);

        timeval_subtract(&t[i], &t2, &t1);

        printf("    Tempo gasto em matrix_dgemm_%d: %lu.%06lus\n",
                i,
                t[i].tv_sec,
                t[i].tv_usec
        );

        if (matrix_eq(N, res[i], C))
            printf("    Resultado OK!\n");
        else
        {
            printf("    Resultado INCORRETO!\n");
            abort();
        }

        if (timeval_cmp(&t[i], previous_time) <= 0)
            printf("    Tempo OK!\n\n");
        else
        {
            printf("    FALHOU: Sua implementação é mais lenta que a anterior!\n\n");
            abort();
        }

        previous_time = &t[i];
    }

    free(A);
    free(B);
    free(C);
    free(D);
}
