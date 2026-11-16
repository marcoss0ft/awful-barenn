#!/bin/bash
#SBATCH --job-name=aps2_tabela
#SBATCH --output=resultados/tabela_final_%j.out
#SBATCH --partition=pluton_gpu     # limite de 1 h (a v1.0 leva ~2 min por execução)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # 8 núcleos P para a versão OpenMP
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:45:00

# Uso: module load cuda/12.8.1 && make gpu && scripts/compilar_tags.sh   (no nó de login)
#      sbatch slurm/tabela_final.slurm
# Mede TODAS as versões (binários de cada tag e o ./gpu da árvore atual) no
# mesmo job e no mesmo nó, para a tabela consolidada do README.
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
hostname
lscpu | grep "Model name"
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

B=bin_tags
echo "================ Corretude (diff contra v0.0-baseline) ================"
$B/v0.0-baseline | grep -v Tempo > resultados/ref_legado.txt
for v in v0.1-cpu-memoria v0.2-cpu-openmp v1.1-cuda-residente v1.2-cuda-shared; do
    echo "--- $v"
    $B/$v | grep -v Tempo | diff resultados/ref_legado.txt - && echo "identica ao legado"
done
echo "--- atual: ./gpu (--fmad=false)"
./gpu | grep -v Tempo | diff resultados/ref_legado.txt - && echo "identica ao legado"

for par in "v0.0-baseline 10" "v0.1-cpu-memoria 10" "v0.2-cpu-openmp 10" \
           "v1.1-cuda-residente 10" "v1.2-cuda-shared 10" "v1.0-cuda-ingenuo 3"; do
    set -- $par
    echo "================ $1 ================"
    scripts/bench.sh $B/$1 $2
done
echo "================ atual: ./gpu (--fmad=false) ================"
scripts/bench.sh ./gpu 10
