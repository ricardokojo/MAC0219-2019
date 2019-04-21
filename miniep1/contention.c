#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <sys/time.h>

// Used by make to substitute @#@IF with. Do not write any other @#@IF in this
// code except by the ones already written
#define IF_GT_MAX if(t->arr[i] > max)

pthread_mutex_t lock; // protects max
double max = -1; // max array value

// A task, containing a pointer to an array and its size
struct task {
        double *arr;
        int size;
};

// Print an error message and exit with failure code
#define DIE(...) { \
        fprintf(stderr, __VA_ARGS__); \
        exit(EXIT_FAILURE); \
}

// A factorial function. Implemented [almost] the worst way possible.
int factorial(int n) {
        int ret = 1;
        for (int i = 2; i <= n; ++i)
                ret *= i;
        return ret;
}

// This is a complete non-sense laborious function for the threads to spend some
// time with. Have fun, threads :)
// ps: Let's even don't care about possible under or overflows nor NaNs.
double a_laborious_function(double x) {
        double a = atan(x) + cos(x);
        double dadx = (1./(pow(x+0.2, 2) + 1.)) - sin(x);
        double k = a + (dadx * pow(x, 12)) / (double)factorial(9);
        double s = 12.12 / (double)factorial(12) / factorial(11);
        k += (pow(x, 5) / (double)factorial(12)) * s;
        k -= sin(dadx) + 2 * M_PI * cos(dadx);
        return k > 0 ? k : -k;
}

// This function receives a struct task and performs some laborious work on each
// element of the task's array. After the work, if the element has greater value
// then the global variable 'max', 'max' is updated with the element's value.
void *thread_work(void *arg) {
        struct task *t = (struct task *)arg;

        for (int i = 0; i < t->size; ++i) {
                t->arr[i] = a_laborious_function(t->arr[i]);
                t->arr[i] = a_laborious_function(t->arr[i]);
                // let's protect ourselfs from a big mess we may have made
                if(isnan(t->arr[i]) || isinf(t->arr[i]))
                        t->arr[i] = rand() / RAND_MAX;

                // Don't modify the next line. It will be replaced at
                // compilation according to the value passed to the makefile
                // parameter 'IF':
                //@#@IF
                {
                        pthread_mutex_lock(&lock);
                        if (t->arr[i] > max)
                                max = t->arr[i];
                        pthread_mutex_unlock(&lock);
                }

        }

        return NULL;
}

void fill_array(double *V, int N) {
        srand(2382); // arbitrary initialization
        for (int i = 0; i < N; ++i)
                V[i] = (double)rand() / RAND_MAX;
}

int main(int argc, char **argv)
{
        pthread_t *threads;
        unsigned num_threads;
        struct task *tasks;
        int N;
        double *V;
        struct timeval start, end;

        // Argument parsing
        if (argc != 3 || sscanf(argv[1], "%u", &num_threads) != 1 ||
            sscanf(argv[2], "%u", &N) != 1) {
                printf("usage: %s <num_threads> <array_size>\n",
                       argv[0]);
                return 1;
        }

        if (N < 0)
            DIE("Vector size overflow\n");

        // Initialize mutex with default attributes
        if(pthread_mutex_init(&lock, NULL))
                DIE("Failed to init mutex\n");

        // Malloc arrays
        if((threads = malloc(num_threads * sizeof(pthread_t))) == NULL)
                DIE("Threads malloc failed\n");
        if((tasks = malloc(num_threads * sizeof(struct task))) == NULL)
                DIE("Tasks malloc failed\n");
        if((V = malloc(N * sizeof(double))) == NULL)
                DIE("V malloc failed\n");

        fill_array(V, N);

        gettimeofday(&start, NULL);

        // Initialize threads with default attributes.
        // The work is being splitted as evenly as possible between threads.
        int threads_with_one_more_work = N % num_threads;
        for (int i = 0; i < num_threads; ++i) {
                int work_size = N / num_threads;
                if (i < threads_with_one_more_work)
                        work_size += 1;
                tasks[i].arr = V + i * work_size;
                tasks[i].size = work_size;
                if(pthread_create(&threads[i], NULL, thread_work, (void *)&tasks[i]))
                        DIE("Failed to create thread %d\n", i)
        }

        // Finish threads and ignore their return values
        for (int i = 0; i < num_threads; ++i) {
                if(pthread_join(threads[i], NULL))
                        DIE("failed to join thread %d\n", i);
        }

        gettimeofday(&end, NULL);
        double elapsed_time = (end.tv_sec - start.tv_sec) +
                              (end.tv_usec - start.tv_usec) / 1000000.0;
        printf("%.4fs\n", elapsed_time);

        // You may print max if you want to take a look
        // printf("max: %lf\n", max);

        if(pthread_mutex_destroy(&lock)) // Destroy mutex
                DIE("Failed to destroy mutex\n");
        free(threads);
        free(tasks);
        return 0;
}
