#include <stdio.h>
#include <stdlib.h>

#define DIE(...) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
}

#define ABS(x) ((x) < 0 ? (-x) : (x))

void cpu_add(REAL *a, REAL *b, REAL *c, unsigned n) {
        for (int i = 0; i < n; ++i)
                c[i] = a[i] + b[i];
}

int check_add(REAL *a, REAL *b, REAL *c, unsigned n) {
        for (int i = 0; i < n; ++i) {
                REAL sum = a[i] + b[i];
                if (ABS(sum - c[i]) > 0.0001)
                        return 0;
        }
        return 1;
}

void random_fill(REAL *a, unsigned n) {
        for (int i = 0; i < n; ++i)
                a[i] = (REAL) rand() / RAND_MAX;
}

int main(int argc, char **argv) {
        unsigned n;
        REAL *a, *b, *c;

        if (argc != 2 || sscanf(argv[1], "%u", &n) != 1)
                DIE("use: %s <array_size>\n", argv[0]);

        srand(1337);

        if ((a = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");
        if ((b = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");
        if ((c = (REAL *)malloc(n * sizeof(REAL))) == NULL)
                DIE("malloc error\n");

        random_fill(a, n);
        random_fill(b, n);

        cpu_add(a, b, c, n);

        if(!check_add(a, b, c, n))
                printf("sum FAILED\n");
        else
                printf("sum correct\n");

        return 0;
}
