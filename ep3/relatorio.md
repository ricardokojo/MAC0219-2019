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

O segundo passo foi fazer a divisão de tarefas usando o **MPI**, que é feita da seguinte maneira:

- Caso haja apenas 1 máquina, o programa é executado da mesma forma que no EP passado;
- Caso haja mais de uma máquina, distribuímos o trabalho por **LINHA**. Ou seja, cada máquina calcula uma linha da imagem. O processo mestre recebe as respostas dos outros processos e também calcula linhas, caso necessário.

Essa forma de separação de trabalho foi escolhido por ser considerada mais intuitiva e mais fácil de controlar, além de combinar com a forma que a imagem é gerada: linha por linha.

## Dificuldades encontradas

A dificuldade inicial foi instalar e testar o **MPI**. Nem todos do grupo conseguiram fazer a biblioteca funcionar em suas próprias máquinas. Assim, os testes foram feitos na máquina **brucutu**, do IME-USP.

A segunda dificuldade foi, assim como no EP passado, compilar e linkar os arquivos necessários. Ficou a dúvida entre usar `gcc`, `g++`, `nvcc`, `mpic++`, quais _flags_ eram necessárias e quais não eram etc. O post no Paca foi de grande ajuda, apesar de ainda ter tido problemas com a flag `-lmpi` na **brucutu**. Para isto, criamos a variável `$(BRUCUTUFLAGS)`, para informar ao compilador o local de `mpi.h`.

## Imagens geradas

todo
