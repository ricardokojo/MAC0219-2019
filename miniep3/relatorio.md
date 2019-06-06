# Relatório - MiniEP3

Ricardo Hideki Hangai Kojo - 10295429

## Solução

**A solução foi fortemente baseada no código disponibilizado pelos monitores no Paca.**

Utilizando a struct `Task`, já disponibilizada, crio um vetor de _Tasks_ separando o intervalo de trabalho para cada processo.

Depois, ao invés de criar _pthreads_ (como no código dos monitores), crio processos usando `fork()`.

Faço `<NUM_PROCESSOS> forks`, sendo que cada filho roda a função `process_work()` (adaptada de `thread_work()` dos monitores), que recebe uma _Task_, calcula seu intervalo e soma à variável `pi_by_4`. Enquanto isso, o pai continua gerando filhos. Ademais, guardo todos os **PIDs** em um vetor.

Após realizar todos os _forks()_, faço uma chamada de `waitpid()` para cada _PID_ guardado.

Ao final, imprimo `pi_by_4 * 4`.

## Desafios encontrados

O primeiro desafio encontrado foi escolher qual biblioteca usar para fazer o compartilhamento de recursos entre processos. Após conversar com outros alunos de Paralela, me recomendaram usar o `mmap`.

Logo, o próximo desafio foi entender e aprender a usar **mmap**. Após estudar um pouco a documentação e alguns exemplos, consegui fazer o **mmap** funcionar, apesar de ter dúvidas sobre as flags a serem utilizadas.

Por último, foi necessário lembrar como usar as chamadas `fork()` e `waitpid()`, que foi resolvido rapidamente após conversar com outros colegas sobre o MiniEP e verificar links que eu havia usado para fazer os EPs de **Sistemas Operacionais**.

## Referências

- https://brennan.io/2015/01/16/write-a-shell-in-c/
- http://man7.org/linux/man-pages/man2/mmap.2.html
- https://www.poftut.com/mmap-tutorial-with-examples-in-c-and-cpp-programming-languages/
- https://linux.die.net/man/2/waitpid
