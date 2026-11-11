#!/bin/bash
#SBATCH --job-name=aps2_gpu
#SBATCH --output=resultados/gpu_%j.out
#SBATCH --partition=pluton_gpu     # limite de 1 h (a versão CUDA ingênua leva ~2 min por execução)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:50:00

# Uso: (no nó de login) module load cuda/12.8.1 && make legado gpu
#      sbatch slurm/gpu.slurm [repeticoes]
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
REPS=${1:-5}

hostname
lscpu | grep "Model name"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "== Saida completa de ./gpu =="
./gpu | tee resultados/saida_gpu.txt
echo "== Diferencas para o legado (linhas de tempo ignoradas) =="
./legado | grep -v Tempo > resultados/saida_legado.txt
grep -v Tempo resultados/saida_gpu.txt | diff resultados/saida_legado.txt - \
    && echo "SAIDA IDENTICA AO LEGADO"
echo "== Benchmark de ./gpu =="
scripts/bench.sh ./gpu "$REPS"
