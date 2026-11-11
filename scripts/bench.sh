#!/bin/bash
# Uso: scripts/bench.sh ./binario [repeticoes]
# Roda o binário N vezes e mostra o "Tempo de treino" impresso pelo programa
# e o tempo total do processo (leitura + treino + avaliação), com média e desvio.
BIN=$1
N=${2:-5}

echo "Binario: $BIN | repeticoes: $N | host: $(hostname)"
for i in $(seq 1 "$N"); do
    inicio=$(date +%s.%N)
    treino=$("$BIN" | grep "Tempo de treino" | awk '{print $4}')
    fim=$(date +%s.%N)
    total=$(awk "BEGIN { print $fim - $inicio }")
    printf "rodada %d | treino %.4f s | total %.4f s\n" "$i" "$treino" "$total"
    echo "$treino $total" >> /tmp/bench_$$.txt
done

awk '{ t += $1; t2 += $1*$1; w += $2; w2 += $2*$2; n++ }
     END {
        mt = t/n; mw = w/n
        st = sqrt((t2 - n*mt*mt)/(n-1)); sw = sqrt((w2 - n*mw*mw)/(n-1))
        printf "MEDIA treino %.4f s (desvio %.4f) | total %.4f s (desvio %.4f)\n", mt, st, mw, sw
     }' /tmp/bench_$$.txt
rm -f /tmp/bench_$$.txt
