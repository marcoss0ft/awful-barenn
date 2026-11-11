#!/bin/bash
#SBATCH --job-name=aps2_escala
#SBATCH --output=resultados/escala_%j.out
#SBATCH --partition=sunny_gpu      # nós com i9-14900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 núcleos P (CPUs 0,2,...,14)
#SBATCH --mem=2G
#SBATCH --time=00:20:00

# Uso: (no nó de login) make legado cpu omp_no_treino
#      sbatch slurm/escala_cpu.slurm
# Mede, no mesmo job, a versão cpu e o experimento de OpenMP no treino com
# 1, 2, 4 e 8 threads, além dos tempos de leitura e de avaliação.
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
hostname
lscpu | grep "Model name"
grep Cpus_allowed_list /proc/self/status

echo "== Validacao com 8 threads =="
export OMP_NUM_THREADS=8
scripts/validar.sh ./legado ./cpu
scripts/validar.sh ./legado ./omp_no_treino

for t in 1 2 4 8; do
    export OMP_NUM_THREADS=$t
    echo "================ ./cpu, $t thread(s) ================"
    # 10 execuções: tempo de treino, de leitura e de avaliação de cada uma
    for r in $(seq 1 10); do ./cpu | grep "Tempo" | tr '\n' ' '; echo; done
    scripts/bench.sh ./cpu 10 | grep MEDIA
done

for t in 1 2 4 8; do
    export OMP_NUM_THREADS=$t
    echo "================ ./omp_no_treino, $t thread(s) ================"
    scripts/bench.sh ./omp_no_treino 3
done
