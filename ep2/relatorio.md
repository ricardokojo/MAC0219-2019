# Relatório EP2 - MAC0219

Bruna Bazaluk - 9797002
Felipe Serras - 8539986
Ricardo Kojo - 10295429

## Coleta de Informações

### Sequencial

### OpenMP

### CUDA

### Comparação

## Explicação da Solução


## Desafios

O primeiro desafio foi lidar com a biblioteca libpng, mas logo aprendemos e começamos a estudar o conjunto de Mandelbrot. Entender todos os seus aspectos e aplicar o algoritmo foi a segunda dificuldade; fizemos primeiro o programa sequencial para testar o algoritmo, que foi usado também na otimização pelo OpenMP e GPU, e não foi trivial criar a imagem correta, mais ainda, foi difícil encontrar números para o teste que gerassem uma imagem interessante, então pesquisamos e falamos com outros colegas e encontramos um intervalo bom.

O que gerou mais dificuldades, como esperado, foi mexer com CUDA. Desde programar (a documentação era muito boa, o que ajudou bastante) até compilar e rodar, e quando finalmente conseguimos, notamos uma leve diferença na escala de cores de imagens geradas pela versão de GPU e CPU para algumas seções do plano complexo específicas. 

Ao investigar, descobrimos que isso acontecia porque o valor da função de mandelbrot para alguns pixels tinha resultados levemente diferentes quando o cálculo era realizado em GPU e CPU. Quando o pixel alterado é um dos maiores do intervalo, o valor do máximo pode mudar, modificando levemente a escala de cores da imagem, já que é calculada em relação ao pixel de valor máximo. Percebemos também que as variáveis destinadas a esses pixels começavam instanciadas corretamente em ambos os casos (CPU e GPU), e eram processadas pelas mesmas operações, entretanto, ainda assim covergiam para valores diferentes ao longo das iterações. 

Apesar de em alguns casos essa propagação gerar uma leve diferença na escala de cores, não se observam praticamente nenhuma diferença entre os padrões das imagens geradas.

Concluímos que esse não é um problema do código e se deve, provavelmente a uma diferença de precisão entre os cálculos da GPU e da CPU, que podem gerar flutuações que se propagam através das iterações.

A respeito da compilação, acabamos pedindo ajuda para os monitores que colocaram instruções no PACA, obrigada.

Apesar de tudo, conseguimos realizar a tarefa proposta e aprendemos muito com os erros e acertos.
