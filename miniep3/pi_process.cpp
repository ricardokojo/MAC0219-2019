#include <iostream>

int main(int argc, char const *argv[])
{
  if (argc != 3)
  {
    std::cout << "usage: pi_process <NUM_PROCESSOS> <NUM_PONTOS>\n";
    return 1;
  }

  unsigned NUM_PROCESSOS;
  unsigned NUM_PONTOS;

  NUM_PROCESSOS = atoi(argv[1]);
  NUM_PONTOS = atoi(argv[2]);

  std::cout << NUM_PROCESSOS << " " << NUM_PONTOS << "\n";

  return 0;
}
