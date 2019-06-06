#include <iostream>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

unsigned NUM_PROCESSOS; // number of worker threads
unsigned NUM_PONTOS;    // number of divisions at integration
double *pi_by_4;        // approximation of pi/4

struct task
{
  int start, end;
};

#define DIE(...)                  \
  {                               \
    fprintf(stderr, __VA_ARGS__); \
    exit(EXIT_FAILURE);           \
  }

void *process_work(void *arg)
{
  struct task *t = (struct task *)arg;
  double acc = 0;                          // Thread's local integration variable
  double interval_size = 1.0 / NUM_PONTOS; // The circle radius is 1.0

  // Integrates f(x) = sqrt(1 - x^2) in [t->start, t->end[
  for (int i = t->start; i < t->end; ++i)
  {
    double x = (i * interval_size) + interval_size / 2;
    acc += sqrt(1 - (x * x)) * interval_size;
  }

  // std::cout << acc << "\n";
  pi_by_4[0] += acc;

  exit(0);
}

int main(int argc, char const *argv[])
{
  struct task *tasks;
  pid_t pid;
  pid_t *list_of_pids;
  int status;

  if (argc != 3)
  {
    std::cout << "usage: pi_process <NUM_PROCESSOS> <NUM_PONTOS>\n";
    return 1;
  }

  NUM_PROCESSOS = atoi(argv[1]);
  NUM_PONTOS = atoi(argv[2]);

  // Malloc tasks
  if ((tasks = (task *)std::malloc(NUM_PROCESSOS * sizeof(struct task))) == NULL)
    DIE("Tasks malloc failed\n");
  if ((list_of_pids = (pid_t *)std::malloc(NUM_PROCESSOS * sizeof(pid_t))) == NULL)
    DIE("list_of_pids malloc failed\n");

  // Separate tasks
  int processes_with_one_more_work = NUM_PONTOS % NUM_PROCESSOS;
  for (int i = 0; i < NUM_PROCESSOS; ++i)
  {
    int work_size = NUM_PONTOS / NUM_PROCESSOS;
    if (i < processes_with_one_more_work)
      work_size += 1;
    tasks[i].start = i * work_size;
    tasks[i].end = (i + 1) * work_size;
  }

  // Create mmap and call forks()
  pi_by_4 = (double *)mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);

  for (int i = 0; i < NUM_PROCESSOS; ++i)
  {
    pid = fork();
    if (pid == 0) // child
    {
      process_work((void *)&tasks[i]);
    }
    else if (pid > 0) // parent
    {
      list_of_pids[i] = pid;
    }
    else // error -> clear memory and die
    {
      free(list_of_pids);
      free(tasks);
      munmap(pi_by_4, sizeof(double));
      DIE("Fork failed\n");
    }
  }

  // Call waitpid() for each PID
  for (int i = 0; i < NUM_PROCESSOS; ++i)
  {
    waitpid(list_of_pids[i], &status, WUNTRACED);
  }

  printf("%.12f\n", pi_by_4[0] * 4);

  free(list_of_pids);
  free(tasks);
  munmap(pi_by_4, sizeof(double));
  return 0;
}
