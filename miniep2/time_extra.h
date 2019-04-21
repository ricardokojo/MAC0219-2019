#include <sys/time.h>


/* Computa result = x - y. retorna 1 se result < 0*/
int timeval_subtract (struct timeval *result, struct timeval *x,
                                              struct timeval *y);

/* Compara x e y.
 * Se x < y, retorna -1
 * Se x > y, retorna  1
 * Se x = y, retorna  0
 */
int timeval_cmp (struct timeval *x, struct timeval *y);
