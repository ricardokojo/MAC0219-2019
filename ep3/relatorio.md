# Relatório EP3 - MAC0219/5742

Bruna Bazaluk - 9797002

Felipe Serras - 8539986

Ricardo Kojo - 10295429

## Explicação da Solução

**Os testes foram feitos nas máquinas dota e brucutuIV do IME-USP.**

A solução encontrada foi fortemente baseada no que foi feito para o EP2.

O primeiro passo foi separar a solução antiga - escrita no arquivo `dmbrot.cu` - em diferentes arquivos:

- `cpu.cpp`: contém funções que calculam e verificam o conjunto de Mandelbrot usando **OpenMP**;
- `gpu.cu:` contém funções que calculam e verificam o conjunto de Mandelbrot usando **CUDA**;
- `img_util.cpp`: contém funções que geram as imagens em _.png_ usando a biblioteca **libpng**;
- `main.cpp`: contém a função `main()`. Lê os argumentos da linha de comando, divide o trabalho (se necessário) e faz as chamadas para fazer os cálculos em CPU ou GPU.

A ideia por trás dessa separação foi encapsular diferentes partes do código que precisassem se utilizar de diferentes bibliotecas, o que permitira compilar cada parte com um programa diferente caso fosse necessário.

O segundo passo foi fazer a divisão de tarefas usando o **MPI**, que é feita da seguinte maneira:

- Caso haja apenas 1 máquina, o programa é executado da mesma forma que no EP passado;
- Caso haja mais de uma máquina, distribuímos o trabalho por **LINHA**. Ou seja, cada máquina calcula um grupo de linhas da imagem. O processo mestre recebe as respostas dos outros processos via **MPI**, os salva e concatena. Além disso ele também calcula linhas remanescentes, caso necessário.

Essa forma de separação de trabalho foi escolhida por ser considerada mais intuitiva e mais fácil de controlar, além de combinar com a forma que a imagem é gerada e salva na memória: linha por linha. No caso do número de processos pedidos ser maior do que o número de linhas, é realizada uma readequação da distribuição do trabalho, usando o máximo de processos possíveis sem corromper as linhas. Entretanto, esse limite dificilmente seria atingido já que trava máxima de processos do mpi é usualmente bem mais baixa que o número de linhas em uma imagem comum.


## Dificuldades encontradas

A dificuldade inicial foi instalar e testar o **MPI**. Nem todos do grupo conseguiram fazer a biblioteca funcionar em suas próprias máquinas. Assim, os testes foram feitos na máquina **brucutuIV**, do IME-USP.

A segunda dificuldade foi, assim como no EP passado, compilar e linkeditar os arquivos necessários. Ficou a dúvida entre usar `gcc`, `g++`, `nvcc`, `mpic++`, quais _flags_ eram necessárias e quais não eram etc. O post no Paca foi de grande ajuda, apesar de ainda ter tido problemas com a flag `-lmpi` na **brucutuIV**. Para resolver esse problema, criamos a variável `$(BRUCUTUFLAGS)`, para informar ao compilador o local de `mpi.h`. Essas flags foram obtidas a partir na opção -showme do mpic++, na qual ele lista todas as flags especiais que está usando para a linkedição naquela máquina.

##  Resultados e imagens geradas

Produzimos algumas imagens de áreas conhecidas do plano complexo e obtivemos resultados iguais aos obtidos no EP anterior. Os tempos também mantiveram padrões parecidos com os do EP2. A adição de mais processos via MPI revelou uma pequena melhora de tempo até certo limite, especialmente para os casos em que os parâmetros do executável não previam paralelização via OpenMP, por exemplo. Entretanto reconhece-se o potencial de melhoria significativa do tempo caso o programa pudesse ser rodado em múltiplas máquinas, via **MPI**, acomodando um número muito maior de processos, e intensificando a paralelização através da distribuição.

A seguir estão algumas imagens de exemplo geradas pelo EP, com seus respectivos parâmetros de entrada.

todo


##Conclusões

Acreditamos ter cumprido com os objetivos do Exercício Programa, distribuindo a tarefa realizada no EP anterior via **MPI**. Apesar de não termos podido testar em um sistema com mais de uma máquina e apesar dos problemas que tivemos com a compilação e linkedição, sentimos que aprendemos muito sobre os conceitos de distribuição e sobre como usar o **MPI** para gerá-la. Sentimos que aprendemos bastante também sobre o processo de compilação, no geral.
