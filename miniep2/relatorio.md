# MAC0219 - MiniEP2

Ricardo Hideki Hangai Kojo - 10295429

Os testes para este MiniEP2 foram feitos em um desktop com:

* **Sistema Operacional** => Ubuntu 19.04 Disco Dingo 64-bit
* **Processador** => Ryzen 5 1600, 6 Cores, 12 Threads, 3.4 Ghz, 16MB de Cache L3, 3MB de Cache L2
* **Memória RAM** => 8 GB Crucial Ballistix Sport LT, 2400MHz, Single-channel

## <a id="implementacao"></a> Implementação

A implementação do **algoritmo 0** se aproxima da forma de como aprendemos a multiplicar matrizes na escola. Fixamos um ponto na matrix `C`, percorremos uma linha da matriz `A` e uma coluna da matriz `B`, fazendo as multiplicações e somas necessárias para calcular o valor na matrix `C`.

No **algoritmo 1**, ao invés de fixar e calcular uma coordenada em `C` de uma vez, fixamos uma coordenada em `A`. Com esta coordenada, percorremos a linha de `B` que a utiliza para calcular o valor em `C`. Desta forma, preenchemos a matriz `C` "aos poucos" e fazemos uso do *cache*, já que lemos a matriz `B` por linhas ao invés de colunas.

Já para o **algoritmo 2**, utilizamos o que foi feito no *algoritmo 1* juntamente com a técnica de *blocagem*. Assim, adicionamos dois *loops `for`* para fazer controle dos blocos. Fora isso, o código é praticamente o mesmo do *algoritmo 1*.

## Desafios

O primeiro desafio foi entender mais sobre *cache*. Devido à alguns compromissos, acabei perdendo duas aulas e não havia visto conteúdo sobre o assunto.

O segundo desafio veio na hora da implementação, que foi a dificuldade em manter o controle dos índices dos *loops*. Ajeitar tudo deu um certo trabalho e vários `segmentation fault`.

## Testes realizados

Para este MiniEP2 foram realizados diversos testes com os diferentes algoritmos, com matrizes de 1024x1024 e 2048x2048, e com diversos tamanhos de bloco:

* A tabela a seguir representa os testes feitos usando o comando `make test`. Ao total, foram feitos 10 testes e foi calculada a média para cada algoritmo:

    | Teste / Algoritmo         |  matrix_dgemm_0 |  matrix_dgemm_1 |  matrix_dgemm_2 |
    |---------:|----------------:|----------------:|----------------:|
    | Teste  1 |     63.697248 s |      6.593700 s |      5.682541 s |
    | Teste  2 |     63.220010 s |      6.234701 s |      5.587355 s |
    | Teste  3 |     60.525148 s |      6.663199 s |      5.610183 s |
    | Teste  4 |     58.635916 s |      5.763930 s |      5.410042 s |
    | Teste  5 |     59.883052 s |      5.866340 s |      5.669662 s |
    | Teste  6 |     62.269344 s |      5.944602 s |      5.394697 s |
    | Teste  7 |     61.187431 s |      5.781536 s |      5.419204 s |
    | Teste  8 |     60.071612 s |      5.569825 s |      5.088236 s |
    | Teste  9 |     58.044204 s |      5.530791 s |      5.118510 s |
    | Teste 10 |     61.107507 s |      6.197545 s |      5.246770 s |
    |   **Média**  |     **60.864147s** |      **6.014617s** |      **5.422720s** |

* A tabela a seguir representa os testes feitos usando o `./main` para matrizes de tamanho {1024x1024, 2048x2048} e algoritmos {0, 1}. Ao total, foram feitos 30 testes de cada, mas a tabela mostra apenas as **médias de tempo**:

    | Tamanho da Matriz / Algoritmo | matrix_dgemm_0 | matrix_dgemm_1 |
    |----------------------:|----------------:|----------------:|
    |    **1024 x 1024**    |     6.6735766 s |      0.711549 s |
    |    **2048 x 2048**    |     63.578609 s |      6.149208 s |

* A tabela a seguir representa os testes feitos usando o `./main` para matrizes de tamanho {1024x1024, 2048x2048}, e algoritmo {2} com {2, 4, 8, 16} blocos. Ao total, foram feitos 30 testes de cada configuração, mas a tabela mostra apenas as **médias de tempo**:

    | Blocos / Tamanho da Matriz |   1024x1024 |   2048x2048 |
    |---------------------------:|------------:|------------:|
    |            **2**           | 0.6606788 s | 7.7288196 s |
    |            **4**           | 0.7238957 s | 5.5499432 s |
    |            **8**           | 0.7782400 s | 5.6493434 s |
    |           **16**           | 0.9078439 s | 6.4793124 s |

## Perguntas

### 1. Mostre, com embasamento estatístico, a variação de tempo entre matrix_dgemm_1 e sua implementação de matrix_dgemm_0. Houve melhora no tempo de execução? Explique porque

Nos testes realizados, *matrix_dgemm_0* demorou, em média:

* **60.864147s** usando `make test`;
* **6.6735766s** usando `./main --matrix-size 1024 --algorithm 0`;
* **63.578609s** usando `./main --matrix-size 2048 --algorithm 0`.

Já o *matrix_dgemm_1* demorou, em média:

* **6.014617s** usando `make test`;
* **0.711549s** usando `./main --matrix-size 1024 --algorithm 1`;
* **6.149208s** usando `./main --matrix-size 2048 --algorithm 1`.

Assim, temos que o algoritmo de *matrix_dgemm_1* foi, aproximandamente:

* **`10.12x` mais rápido** que o algoritmo *matrix_dgemm_0* para `make test`;
* **`9.34x` mais rápido** que o algoritmo *matrix_dgemm_0* para `./main --matrix-size 1024 --algorithm 1`;
* **`10.34x` mais rápido** que o algoritmo *matrix_dgemm_0* para `./main --matrix-size 2048 --algorithm 1`;

A melhora no tempo de execução é evidente (aprox. 10 vezes mais rápido), e decorre do uso do *cache*. Como foi brevemente explicado na seção [Implementação](#implementacao), a mudança feita no algoritmo permite que haja *chace hits* quando lemos a matrix `B`, pois estamos lendo por linhas ao invés de colunas.

### 2. Mostre, com embasamento estatístico, a variação de tempo entre matrix_dgemm_2 e sua implementação de matrix_dgemm_1. Houve melhora no tempo de execução? Explique porque

Nos testes realizados, *matrix_dgemm_1* demorou, em média:

* **6.014617s** usando `make test`;
* **0.711549s** usando `./main --matrix-size 1024 --algorithm 1`;
* **6.149208s** usando `./main --matrix-size 2048 --algorithm 1`.

Já o *matrix_dgemm_2* demorou, em média:

* **5.422720s** usando `make test`;
* **0.6606788s** usando `./main --matrix-size 1024 --algorithm 2` com 2 blocos (melhor tempo dentre os testes);
* **5.5499432s** usando `./main --matrix-size 2048 --algorithm 2` com 4 blocos (melhor tempo dentre os testes).

Assim, temos que o algoritmo de *matrix_dgemm_2* foi, aproximandamente:

* **`1,101x` mais rápido** que o algoritmo *matrix_dgemm_1* para `make test`;
* **`1,077x` mais rápido** que o algoritmo *matrix_dgemm_1* para `./main --matrix-size 1024 --algorithm 1` com 2 blocos;
* **`1,108x` mais rápido** que o algoritmo *matrix_dgemm_1* para `./main --matrix-size 2048 --algorithm 1` com 4 blocos;

Comparado à questão anterior, a melhora no tempo de execução foi bem menor (cerca de 10% mais rápido). Esse aumento se deve à técnica de *blocagem*, que permite que haja mais *cache hits* usando o princípio da localidade. Ao percorrer pequenas partes das linhas (ou seja, ao percorrer os blocos), há maior chance de *cache hits* comparado à ler a linha completa (principalmente em matrizes grandes).

### 3. Como você usou a blocagem para melhorar a velocidade da multiplicação de matrizes?

Para melhorar a velocidade foi necessário testar diferentes números de blocos. Ter blocos demais ou blocos de menos pode afetar o tempo de execução negativamente, como foi visto nos testes.

Por exemplo:

* Para matrizes **1024x1024**, a melhor quantidade de blocos foi `2`. Conforme a *quantidade de blocos aumentava*, o *tempo de execução também aumentava*.
* Para matrizes **2048x2048**, a melhor quantidade de blocos ficou entre `4` e `8`. Neste exemplo, apenas `2` blocos aumentava o tempo de execução, da mesma forma que `16` blocos.

No entanto, isso funcionou apenas para matrizes *1024x1024* e *2048x2048*. Por curiosidade, testei com matrizes *512x512* e *4096x4096*, e não consegui fazer com que o *algoritmo 2* fosse melhor que o *1*, mesmo testando com diferentes quantidades de blocos.