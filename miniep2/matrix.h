/* Operações com matrizes. */

/* Isso aqui embaixo é um header guard. Serve para evitar dupla-inclusão
 * deste cabeçalho
 */
#ifndef _MATRIX_H
#define _MATRIX_H

/* Para usar booleanos em C */
#include <stdbool.h>

#define FOR_EACH_DGEMM(i) for (i = 0; i < 3; ++i)

/* Encha a matriz A com dados aleatórios.
 * A: Matriz
 * n: número de linhas = número de colunas
 */
void matrix_fill_rand(unsigned n, double *restrict _A);

/* Computa C = A*B
 * Algoritmo naive,
 */
void matrix_dgemm_0(unsigned n, double *restrict _C, double *restrict _A,
                                                     double *restrict _B);
/* Computa C = A*B
 * Seu algoritmo
 */
void matrix_dgemm_1(unsigned n, double *restrict _C, double *restrict _A,
                                                     double *restrict _B);

/* Computa C = A*B
 * Seu algoritmo, com blocking
 */
void matrix_dgemm_2(unsigned n, double *restrict _C, double *restrict _A,
                                                     double *restrict _B);

/* Seleciona qual dgemm usar */
bool matrix_which_dgemm(int algorithm, unsigned n, double *restrict _C,
                        double *restrict _A, double *restrict _B);

/* Retorna true se as matrizes são iguais, e false c.c.*/
bool matrix_eq(unsigned n, double *restrict _A, double *restrict _B);

#endif
