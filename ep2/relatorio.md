# Relatório EP2 - MAC0219

Bruna Bazaluk - 9797002

Felipe Serras - 8539986

Ricardo Kojo - 10295429

## Coleta de Informações

Os testes para o EP2 foram realizados utilizando a máquina `dota` do IME-USP, via `ssh`.

### Sequencial

Para verificar o algoritmo sequencial, foram feitos os seguintes testes:

| Tamanho da Matriz | Tempo médio | int max  | int min
| ----------------: | ----------: | -------:  | --------:
|         1024x1024 |    21.3754s | 21.3966  | 21.3542
|         2048x2048 |    84.9706s |85.0355  | 84.9056
|         4096x4096 |   340.7568s |341.0329 |340.4807
|         8192x8192 |   1360.418s |1360.4954 | 1360.3406

O intervalo de confiança é de **95%**.


Para as matrizes de tamanho *1024x1024* e *2048x2048* foram realizados **10** testes cada. Para matriz de tamanho *4096x4096*, foram realizados **5** testes. Para matriz de tamanho *8192x8192*, foram realizados **3** testes.

### OpenMP

Para verificar o algoritmo utilizando **OpenMP**, foram feitos os seguintes testes:

| Threads / Tamanho da Matriz | 1024x1024 | 2048x2048 | 4096x4096 | 8192x8192 | 
| --------------------------: | --------: | --------: | --------: | --------: |
|                           2 |    10.82s |  43.0185s | 176.1984s |  691.563s |
|       Intervalo de confiança    |[10.8008,10.8392] | [,43.0017,43.0354] | [176.1630,176.2338] | [689.9426,693.1834]
|                           4 |   8.9343s |  35.6331s |  141.242s |  577.898s |
|Intervalo de confiança |                             [,8.9245,8.9441 ]|[35.6233,35.6429]|[141.2116,141.2724]|[575.9827,579.8147]
|                           8 |   5.5181s |  21.8835s |  86.5784s |  363.174s |
|Intervalo de confiança |                            [ 5.5115,5.5247]|[21.8723,21.8946]|[86.5457,86.6111]|[362.3140,364.0340]
|                          16 |   3.1523s |  12.4134s |  47.6483s |  197.799s |
|Intervalo de confiança|                          [3.1401,3.1645	]|[12.3797,12.4472]|[47.601247.6954]|[196.9795,198.6185]

Os valores na tabela representam o tempo médio retirado após 10 testes para cada combinação.

### CUDA

Para verificar o algoritmo utilizando o **CUDA**, foram feitos os seguintes testes:

| Threads por bloco / Tamanho da Matriz | 1024x1024 | 2048x2048 | 4096x4096 | 8192x8192 |
| ------------------------------------: | --------: | --------: | --------: | --------: |
|                                    32 |    3.518s |   3.2431s |   4.3964s |   6.8959s |
|Intervalo de confiança|      [3.5021,3.5338] | [3.2311,3.2551]|[4.4168,4.4168]|[6.7977,6.9941]
|                                    64 |   3.5156s |   3.2201s |   4.2585s |    5.804s |
|Intervalo de confiança|[3.4938,3.5373]|[3.2091,3.2311]|[4.2418,4.2752]|[5.6894,5.9186]
|100 |   3.4771s |   3.2337s |    4.2844 |   5.5748s |
|Intervalo de confiança| [3.4536,3.5006]|[3.2078,3.2596]|[4.2416,4.3271]|[5.5416,5.6080]
|                                   128 |    3.498s |   3.3085s |   4.6489s |   6.1702s |
|Intervalo de confiança|[3.4795,3.5168]|[3.2856,3.3314]|[4.6182,4.6796]|[6.0538,6.2866]
|                                   256 |   3.5196s |   4.0145s |   4.6415s |   5.9616s |
|Intervalo de confiança|[3.5014,3.5378]|[3.7230,4.3060]|[4.6180,4.6650]|[5.8096,6.1135]
### Comparação

Observamos que os tempos de execução sobem muito mais rápido no algoritmo sequencial, e extremamente devagar no algoritmo em CUDA. Construímos um gráfico para ilustrar, os dados de OpenMP foram retirados das medidas de 16 threads, e os de CUDA 256. 
INSERE AQUI O GRAFICO_SHOW
Apesar de algumas flutuações, o tempo tende a crescer junto com o tamanho da imagem, nos três algoritmos. Nos algoritmos paralelos, a quantidade de threads também influencia no tempo de execução.


## Explicação da Solução

Para criar o algoritmo da nossa solução, seguimos o que foi explicado no enunciado. Baseado no esquema (Figura 2), simulamos a matriz e, para cada ponto da matriz, calculamos a sequência para `1000` iterações, verificando se o ponto pertencia ao conjunto de Mandelbrot. Para fazer cálculos com números complexos, utilizamos a biblioteca `thrust::complex`.

Guardávamos os valores necessários para a geração da imagem no vetor `buffer_image` que simulava a matriz. Caso o valor pertencesse ao conjunto, guardávamos o valor `0`. Caso não pertencesse, o vetor recebia o número da iteração em que a condição `|z_j| <= 2` foi excedida.

Após calcular os valores para todos os pontos da matriz, normalizamos o vetor `buffer_image` e geramos a imagem utilizando a biblioteca `libpng`.

Para as versões em OpenMP e CUDA, o algoritmo é praticamente o mesmo, fazendo apenas os ajustes necessários para fazer uso destas tecnologias. Para OpenMP, foram colocados dois `pragmas`: um no loop que percorre a matriz e calcula a sequência, e outro no loop que normaliza o vetor `buffer_image`. Para CUDA, usamos o `buffer_image` como variável compartilhada, fazendo tanto o cálculo da sequência quanto a normalização em GPU.

## Desafios

O primeiro desafio foi lidar com a biblioteca libpng, mas logo aprendemos e começamos a estudar o conjunto de Mandelbrot. Entender todos os seus aspectos e aplicar o algoritmo foi a segunda dificuldade; fizemos primeiro o programa sequencial para testar o algoritmo, que foi usado também na otimização pelo OpenMP e GPU, e não foi trivial criar a imagem correta, mais ainda, foi difícil encontrar números para o teste que gerassem uma imagem interessante, então pesquisamos e falamos com outros colegas e encontramos um intervalo bom.

O que gerou mais dificuldades, como esperado, foi mexer com CUDA. Desde programar (a documentação era muito boa, o que ajudou bastante) até compilar e rodar, e quando finalmente conseguimos, notamos uma leve diferença na escala de cores de imagens geradas pela versão de GPU e CPU para algumas seções do plano complexo específicas.

Ao investigar, descobrimos que isso acontecia porque o valor da função de mandelbrot para alguns pixels tinha resultados levemente diferentes quando o cálculo era realizado em GPU e CPU. Quando o pixel alterado é um dos maiores do intervalo, o valor do máximo pode mudar, modificando levemente a escala de cores da imagem, já que é calculada em relação ao pixel de valor máximo. Percebemos também que as variáveis destinadas a esses pixels começavam instanciadas corretamente em ambos os casos (CPU e GPU), e eram processadas pelas mesmas operações, entretanto, ainda assim covergiam para valores diferentes ao longo das iterações. 

Apesar de em alguns casos essa propagação gerar uma leve diferença na escala de cores, não se observam praticamente nenhuma diferença entre os padrões das imagens geradas.

Concluímos que esse não é um problema do código e se deve, provavelmente a uma diferença de precisão entre os cálculos da GPU e da CPU, que podem gerar flutuações que se propagam através das iterações.

A respeito da compilação, acabamos pedindo ajuda para os monitores que colocaram instruções no PACA, obrigada.

Apesar de tudo, conseguimos realizar a tarefa proposta e aprendemos muito com os erros e acertos.
