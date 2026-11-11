#!/bin/bash
#SBATCH --job-name=aps2_cpu
#SBATCH --output=resultados/cpu_%j.out
#SBATCH --partition=sunny_gpu      # nós com i9-14900 (mesma máquina dos testes de GPU)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=00:15:00

# Uso:  make legado <alvo>          (no nó de login: os nós de computação não
#                                    têm os headers da glibc para compilar)
#       sbatch slurm/cpu.slurm ./<binario> [repeticoes]
# Sem argumentos, mede o legado -O0 (contexto) e o legado -O3 (baseline).

# Os nós de computação não têm a libstdc++ do gcc 14 no caminho padrão
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

BIN=${1:-}
REPS=${2:-10}
lscpu | grep "Model name"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

if [ -z "$BIN" ]; then
    echo "== Legado -O0 (contexto) =="
    scripts/bench.sh ./legado_O0 3
    echo "== Legado -O3 (baseline) =="
    scripts/bench.sh ./legado "$REPS"
else
    echo "== Validacao de $BIN contra o legado =="
    scripts/validar.sh ./legado "$BIN"
    echo "== Saida completa de $BIN =="
    "$BIN"
    echo "== Benchmark de $BIN =="
    scripts/bench.sh "$BIN" "$REPS"
fi
