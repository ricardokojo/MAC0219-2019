#include <iostream>
#include <math.h>

unsigned NUM_PROCESSOS; // number of worker threads
unsigned NUM_PONTOS;    // number of divisions at integration
double pi_by_4 = 0;     // approximation of pi/4

struct task
{
  int start, end;
};

void *process_word(void *arg)
{
  struct task *t = (struct task *)arg;
  double acc = 0;                 // Thread's local integration variable
  double interval_size = 1.0 / N; // The circle radius is 1.0

  // Integrates f(x) = sqrt(1 - x^2) in [t->start, t->end[
  for (int i = t->start; i < t->end; ++i)
  {
    double x = (i * interval_size) + interval_size / 2;
    acc += sqrt(1 - (x * x)) * interval_size;
  }

  // This is a critical section. As we are going to write to a global
  // value, the operation must me protected against race conditions.
  // pthread_mutex_lock(&lock);
  pi_by_4 += acc;
  // pthread_mutex_unlock(&lock);

  return NULL;
}

int main(int argc, char const *argv[])
{
  struct task *tasks;

  if (argc != 3)
  {
    std::cout << "usage: pi_process <NUM_PROCESSOS> <NUM_PONTOS>\n";
    return 1;
  }

  NUM_PROCESSOS = atoi(argv[1]);
  NUM_PONTOS = atoi(argv[2]);

  std::cout << NUM_PROCESSOS << " " << NUM_PONTOS << "\n";

  if ((tasks = std::malloc(NUM_PROCESSOS * sizeof(struct task))) == NULL)
    DIE("Tasks malloc failed\n");

  // Initialize threads with default attributes.
  // The work is being splitted as evenly as possible between threads.
  int threads_with_one_more_work = NUM_PONTOS % NUM_PROCESSOS;
  for (int i = 0; i < NUM_PROCESSOS; ++i)
  {
    int work_size = NUM_PONTOS / NUM_PROCESSOS;
    if (i < threads_with_one_more_work)
      work_size += 1;
    tasks[i].start = i * work_size;
    tasks[i].end = (i + 1) * work_size;

    // fork()

    // if (pthread_create(&threads[i], NULL, thread_work, (void *)&tasks[i]))
    // DIE("Failed to create thread %d\n", i)
  }

  // Finish threads and ignore their return values
  for (int i = 0; i < num_threads; ++i)
  {
    if (pthread_join(threads[i], NULL))
      DIE("failed to join thread %d\n", i);
  }

  printf("pi ~= %.12f\n", pi_by_4 * 4);

  free(tasks);
  return 0;
}
