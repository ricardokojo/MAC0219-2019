/* Arquivo principal.
 * Programa criado para facilitar a coleta de dados a partir de um script.
 * Para pegar o tempo, basta ler o número impresso pelo stdout.
 */

#include "matrix.h"
#include "time_extra.h"

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define INVALID_ALG  3 /* Parâmetro algorithm não inicializado */
#define INVALID_SIZE 0 /* Tamanho da matriz não incializado */

struct parsed_args
{
    unsigned matrix_size; /* tamanho da matriz */
    unsigned algorithm;   /* qual dgemm vai ser chamado? */
};


/* Parseie os argumentos fornecidos pelo argv */
static struct parsed_args* parse_args(int argc, const char* argv[])
{
    /* A keyword `static` faz com que o compilador aloque o conteúdo no
     * segmento .data, assim o dado não será destruido quando a função
     * retornar. Note também que isso não é thread-safe.
     */
    static struct parsed_args ret = {
        .matrix_size = INVALID_SIZE,
        .algorithm = INVALID_ALG
    };

    int i;

    for (i = 0; i < argc-1; ++i)
    {
        if (!strcmp(argv[i], "--matrix-size"))
            ret.matrix_size = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--algorithm"))
            ret.algorithm = atoi(argv[i+1]);
    }
    return &ret;
}

/* Checa se os argumentos parseados são válidos */
static bool validate_args(const struct parsed_args* args)
{
    if (args->matrix_size == INVALID_SIZE || args->algorithm >= INVALID_ALG)
        return false;
    return true;
}

static void print_usage_message()
{
    const char* msg =
    "Parâmetros incorretos. Uso:\n"
    "  main <ARGS>\n"
    "onde:\n"
    "  --matrix-size <NUM>     Tamanho da matriz quadrada. (n_linhas = n_colunas)\n"
    "  --algorithm <NUM>       Número da implementação. (0 = dgemm_0, 1 = dgemm_1, 2 = dgemm_2)\n"
    "\n";

    printf(msg);
}

int main(int argc, const char* argv[])
{
    struct parsed_args* args = parse_args(argc, argv);

    /* Aqui, restrict quer dizer que o apontador não terá "aliasing".
     * Procure por "pointer aliasing" no google.
     */
    double *restrict A; /* Matriz A */
    double *restrict B;
    double *restrict C;

	struct timeval t1, t2, t3;

    unsigned n;

    if (!validate_args(args))
    {
        print_usage_message();
        return 1;
    }

    n = args->matrix_size;

    A = aligned_alloc(8, n*n*sizeof(*A)); /* Aloque memória com alinhamento de 8 bytes */
    B = aligned_alloc(8, n*n*sizeof(*B));
    C = aligned_alloc(8, n*n*sizeof(*C));

    if (!(A && B && C))
    {
        printf("Seu computador não tem memória pra isso!\n");
        return 2;
    }

    matrix_fill_rand(n, A);
    matrix_fill_rand(n, B);

    memset(C, 0x00, n*n*sizeof(*A));

    gettimeofday(&t1, NULL);
    if (!matrix_which_dgemm(args->algorithm, n, C, A, B))
    {
        printf("Erro: O algoritmo selecionado é invalido!\n");
        return 3;
    }
    gettimeofday(&t2, NULL);

    timeval_subtract(&t3, &t2, &t1);

    printf("%lu.%06lu\n", t3.tv_sec, t3.tv_usec);

    free(A);
    free(B);
    free(C);

    return 0;
}
