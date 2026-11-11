# APS 2: otimização e migração para CUDA do Barenn (sumário executivo)

Rede densa 64 → 32 → 16 → 10 treinada por SGD **uma amostra por vez** (`optdigits.tra`, 3823 imagens × 51 épocas; avaliação em `optdigits.tes`, 1797 imagens). Todas as versões estão no mesmo repositório:

| Arquivo | O que é |
|---|---|
| `barenn.cpp` | código legado, intacto |
| `barenn_cpu.cpp` | versão CPU otimizada |
| `barenn_gpu.cu` | versão CUDA |

Cada degrau de desempenho é uma tag anotada ligada a uma Issue.

## Tabela consolidada

Todas as versões foram compiladas **a partir das próprias tags** (`scripts/compilar_tags.sh`) e medidas **no mesmo job e no mesmo nó** (`slurm/tabela_final.slurm`, job 30340).
- **Máquina:** Franky `compute22`, Intel Core i9-14900 (8 núcleos P alocados) + NVIDIA RTX A1000 (Ampere cc 8.6, 18 SMs).
- **Compiladores:** `g++ -O3` (14.2.0) e `nvcc -O3` (CUDA 12.8.1).
- **Repetições:** 10 por versão (3 para a CUDA inicial). Log em [`resultados/final/tabela_final.txt`](resultados/final/tabela_final.txt).

| Versão (tag) | Técnica principal | Treino (média ± desvio) | Speedup do treino vs legado | Processo completo | Saída |
|---|---|---|---|---|---|
| **Baseline legado** ([`v0.0-baseline`](https://github.com/marcoss0ft/awful-barenn/tree/v0.0-baseline)) | código original, `g++ -O3` | 0,3723 s ± 0,0013 | 1,00× | 0,4003 s | referência |
| CPU: memória ([`v0.1-cpu-memoria`](https://github.com/marcoss0ft/awful-barenn/tree/v0.1-cpu-memoria)) | pesos contíguos, sem cópias/`malloc` por amostra, sem gradiente descartado | 0,2219 s ± 0,0022 | 1,68× | 0,2498 s | idêntica |
| **CPU otimizada** ([`v0.2-cpu-openmp`](https://github.com/marcoss0ft/awful-barenn/tree/v0.2-cpu-openmp)) | + OpenMP (8 threads) na leitura e na avaliação | **0,2218 s ± 0,0021** | **1,68×** | **0,2319 s (1,73×)** | idêntica |
| **CUDA inicial** ([`v1.0-cuda-ingenuo`](https://github.com/marcoss0ft/awful-barenn/tree/v1.0-cuda-ingenuo)) | kernels por camada, `cudaMalloc`/`cudaMemcpy` a cada chamada | 124,53 s ± 0,28 | 0,003× | 127,26 s | equivalente* |
| CUDA residente ([`v1.1-cuda-residente`](https://github.com/marcoss0ft/awful-barenn/tree/v1.1-cuda-residente)) | dados e parâmetros residentes na GPU | 6,10 s ± 0,13 | 0,06× | 8,24 s | equivalente* |
| **CUDA otimizado** ([`v1.2-cuda-shared`](https://github.com/marcoss0ft/awful-barenn/tree/v1.2-cuda-shared)) | 1 kernel por época, pesos em **memória compartilhada** com passo anti-conflito de banco | **0,3950 s ± 0,0079** | 0,94× | 2,55 s | equivalente* |

\* Perdas iguais até a 3ª casa. Na 4ª casa diferem nas épocas 41/46 (0,0156/0,0144 contra 0,0155/0,0145), e a acurácia é 95,6% contra 95,5% (1 imagem). O PTX da GPU usa FMA (`fma.rn.f32`, um arredondamento) e o binário da CPU não (`mulss` + `addss`, dois arredondamentos). As três versões CUDA dão exatamente a mesma saída entre si. Detalhes na [Issue #7](https://github.com/marcoss0ft/awful-barenn/issues/7).

**Na GPU**, a evolução foi 124,5 s → 6,1 s → 0,40 s (**315×** da CUDA inicial para a otimizada). A ablação no mesmo kernel ([Issue #11](https://github.com/marcoss0ft/awful-barenn/issues/11)):

| Variante do kernel | Treino |
|---|---|
| pesos na memória global | 0,93 s |
| pesos em memória compartilhada | 0,54 s |
| memória compartilhada + passo 65 | 0,40 s |

### Leitura dos resultados
- **CPU.** O ganho veio de **eliminar trabalho**: o gradiente da camada 1 era calculado e descartado, e havia ≈ 15 `malloc`/cópias por amostra. Instruções caíram 42% e leituras 56% (callgrind). Misses de cache são ≈ 0% antes e depois, porque os pesos (11 KB) cabem no L1.
- **Paralelismo multicore.** O treino é serial: cada amostra usa os pesos da anterior. OpenMP acelera só leitura (4,7×) e avaliação. O processo fica limitado pela **Lei de Amdahl** (p ≈ 11% → máximo 1,11×). OpenMP dentro do treino fica de 2,3× a 40× **mais lento**, porque ≈ 0,9 µs de trabalho por amostra não pagam 8 barreiras.
- **GPU.** A versão ingênua passava ≈ 95% do tempo em `cudaMalloc`/`cudaMemcpy`/`cudaFree`, com cópias de 0,9 KB a 0,19 GB/s num PCIe Gen4 ×16: latência, não banda. Manter os dados na GPU deu 20×. Fundir a época num kernel e usar memória compartilhada deu mais 15×.
- **Por que a GPU não vence a CPU aqui.** O SGD por amostra deixa só ≈ 32 threads úteis em 1 de 18 SMs (ocupação teórica de 8,3% de um SM e 0,46% da GPU), e cada neurônio é uma cadeia serial de 64 FMAs. O kernel roda a ≈ 5,9 GFLOP/s, cerca de 1% do pico de um SM: é limitado por **latência**, não por cálculo nem por banda. A CPU (5,5 GHz, L1 de ≈ 5 ciclos) percorre essa cadeia mais rápido. A GPU só compensaria com mais trabalho independente por passo (redes mais largas ou mini-batch, que muda o algoritmo).

## Índice dos blocos lógicos

| # | Bloco | Issue (hipótese, dados e discussão) | Pull Request | Tag |
|---|---|---|---|---|
| 0 | Baseline reprodutível e perfil do legado | [#1](https://github.com/marcoss0ft/awful-barenn/issues/1) | [#2](https://github.com/marcoss0ft/awful-barenn/pull/2) | [`v0.0-baseline`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v0.0-baseline) |
| 1 | CPU: layout contíguo, sem cópias e sem trabalho descartado | [#3](https://github.com/marcoss0ft/awful-barenn/issues/3) | [#4](https://github.com/marcoss0ft/awful-barenn/pull/4) | [`v0.1-cpu-memoria`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v0.1-cpu-memoria) |
| 2 | CPU: OpenMP na leitura e na avaliação (+ experimento negativo no treino) | [#5](https://github.com/marcoss0ft/awful-barenn/issues/5) | [#6](https://github.com/marcoss0ft/awful-barenn/pull/6) | [`v0.2-cpu-openmp`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v0.2-cpu-openmp) |
| 3 | GPU: migração funcional para CUDA | [#7](https://github.com/marcoss0ft/awful-barenn/issues/7) | [#8](https://github.com/marcoss0ft/awful-barenn/pull/8) | [`v1.0-cuda-ingenuo`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v1.0-cuda-ingenuo) |
| 4 | GPU: dados residentes na GPU | [#9](https://github.com/marcoss0ft/awful-barenn/issues/9) | [#10](https://github.com/marcoss0ft/awful-barenn/pull/10) | [`v1.1-cuda-residente`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v1.1-cuda-residente) |
| 5 | GPU: kernel por época + memória compartilhada | [#11](https://github.com/marcoss0ft/awful-barenn/issues/11) | [#12](https://github.com/marcoss0ft/awful-barenn/pull/12) | [`v1.2-cuda-shared`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v1.2-cuda-shared) |
| 6 | Relatório: tabela consolidada e este índice | [#13](https://github.com/marcoss0ft/awful-barenn/issues/13) | [#14](https://github.com/marcoss0ft/awful-barenn/pull/14) | [`v1.3-relatorio`](https://github.com/marcoss0ft/awful-barenn/releases/tag/v1.3-relatorio) |

As evidências brutas de cada bloco (benchmarks, callgrind, Nsight Systems) estão em [`resultados/`](resultados/).

## Como reproduzir (cluster Franky)

```bash
git clone https://github.com/marcoss0ft/awful-barenn.git && cd awful-barenn
git checkout <tag>                        # qualquer tag da tabela
module load cuda/12.8.1                   # compile no nó de login
make legado cpu gpu
sbatch slurm/comparar.slurm 10 ./legado ./cpu ./gpu      # valida e mede no mesmo job
scripts/compilar_tags.sh && sbatch slurm/tabela_final.slurm   # tabela acima (a partir da v1.3)
```

Observações do ambiente:
- os nós de computação não compilam (faltam os headers da glibc), por isso a compilação é no nó de login;
- os jobs exportam `LD_LIBRARY_PATH` para a `libstdc++` do gcc 14;
- `valgrind` só existe no nó de login (`scripts/profile_cpu.sh`);
- o Nsight Compute está bloqueado (`ERR_NVGPUCTRPERM`), por isso a ocupação foi calculada a partir da configuração de lançamento.

---

# Enunciado original

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
