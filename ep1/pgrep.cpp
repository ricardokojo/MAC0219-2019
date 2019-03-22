#include <iostream>
#include <pthread.h>

using namespace std;

void pgrep() {
  cout << "Pão\n";
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    cout << "usage: pgrep <MAX_THREADS> <REGEX_PESQUISA> <CAMINHO_DO_DIRETORIO>\n";
    return 1;
  }

  int MAX_THREADS;
  string REGEX, PATH;

  MAX_THREADS = atoi(argv[1]);
  REGEX = argv[2];
  PATH = argv[3];

  cout << "MAX_THREADS: " << MAX_THREADS << "\n";
  cout << "REGEX: " << REGEX << "\n";
  cout << "PATH: " << PATH << "\n";

  pgrep();

  return 0;
}