#!/bin/bash
#SBATCH --job-name=aps2_comparar
#SBATCH --output=resultados/comparar_%j.out
#SBATCH --partition=sunny_gpu      # nós com i9-14900 + RTX A1000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:20:00

# Uso: (no nó de login) make legado <alvos>
#      sbatch slurm/comparar.slurm <repeticoes> ./legado ./bin1 ./bin2 ...
#
# Compara várias versões NA MESMA ALOCAÇÃO (mesmo nó, mesmos núcleos), uma após
# a outra. Entre jobs diferentes o tempo do mesmo binário variou até ~16%
# (turbo de 5,5 a 5,8 GHz nos núcleos P do i9-14900), então só comparamos
# números medidos dentro do mesmo job.
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}

REPS=$1
shift
hostname
lscpu | grep "Model name"
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=name --format=csv,noheader
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

for BIN in "$@"; do
    echo "================ $BIN ================"
    scripts/validar.sh ./legado "$BIN"
    scripts/bench.sh "$BIN" "$REPS"
done
