#!/bin/bash
#SBATCH --job-name=aps2_omp
#SBATCH --output=resultados/omp_intercalado_%j.out
#SBATCH --partition=sunny_gpu      # nós com i9-14900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 núcleos P (CPUs 0,2,...,14)
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# Uso: (no nó de login)
#      git show v0.1-cpu-memoria:barenn_cpu.cpp > /tmp/cpu_v01.cpp
#      g++ -std=c++17 -O3 -Wall /tmp/cpu_v01.cpp -o cpu_v01 && make cpu
#      sbatch slurm/omp_intercalado.slurm
#
# Roda as configurações INTERCALADAS (v01, 1 thread, 8 threads, 8 threads com
# afinidade), 10 rodadas, para que a variação de frequência do nó afete todas
# igualmente. Imprime treino, leitura, avaliação e tempo total do processo.
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
hostname
grep Cpus_allowed_list /proc/self/status
lscpu -e=CPU,CORE,MAXMHZ | head -9

for r in $(seq 1 10); do
    for cfg in v01 t1 t8 t8bind; do
        unset OMP_PROC_BIND OMP_PLACES
        BIN=./cpu
        case $cfg in
            v01)    BIN=./cpu_v01; export OMP_NUM_THREADS=1 ;;
            t1)     export OMP_NUM_THREADS=1 ;;
            t8)     export OMP_NUM_THREADS=8 ;;
            t8bind) export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ;;
        esac
        inicio=$(date +%s.%N)
        tempos=$($BIN | grep Tempo | awk '{print $4}' | tr '\n' ' ')
        fim=$(date +%s.%N)
        # colunas: config treino [leitura avaliacao] total
        echo "$cfg $tempos$(awk "BEGIN { print $fim - $inicio }")"
    done
done > resultados/omp_intercalado_dados.txt
cat resultados/omp_intercalado_dados.txt

echo "== Medias (10 rodadas) =="
awk '{ n[$1]++; tr[$1] += $2; tot[$1] += $NF; tot2[$1] += $NF * $NF
       if (NF == 5) { le[$1] += $3; av[$1] += $4 } }
     END { for (k in n) {
             m = tot[k] / n[k]; s = sqrt((tot2[k] - n[k] * m * m) / (n[k] - 1))
             printf "%-7s treino %.4f s | leitura %.4f s | avaliacao %.5f s | total %.4f s (desvio %.4f)\n",
                    k, tr[k] / n[k], le[k] / n[k], av[k] / n[k], m, s } }' \
    resultados/omp_intercalado_dados.txt | sort
