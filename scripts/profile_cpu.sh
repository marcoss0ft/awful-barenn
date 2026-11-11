#!/bin/bash
# Uso: scripts/profile_cpu.sh <alvo_do_make>   (ex.: legado_prof)
# Roda no nó de login do Franky: o valgrind só está instalado nele.
# O callgrind simula a execução (conta instruções e, com --cache-sim=yes,
# acessos e misses de cache L1/LL), então o resultado não depende de outros
# processos rodando na máquina.
ALVO=${1:-legado_prof}
make "$ALVO" || exit 1
mkdir -p resultados

nice valgrind --tool=callgrind --cache-sim=yes \
    --callgrind-out-file=resultados/callgrind."$ALVO".out ./"$ALVO" \
    > resultados/callgrind."$ALVO".stdout 2> resultados/callgrind."$ALVO".log

# Resumo por função: instruções (Ir), leituras (Dr), escritas (Dw) e misses
callgrind_annotate --inclusive=no resultados/callgrind."$ALVO".out \
    | head -45 > resultados/callgrind."$ALVO".txt
cat resultados/callgrind."$ALVO".txt
