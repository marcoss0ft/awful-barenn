#!/bin/bash
#SBATCH --job-name=aps2_tags
#SBATCH --output=resultados/testar_tags_%j.out
#SBATCH --partition=pluton_gpu     # limite de 1 h (a v1.0 leva ~2 min por execução)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:45:00

# Uso: module load cuda/12.8.1 && scripts/testar_tags.sh   (no nó de login)
#      sbatch slurm/testar_tags.slurm
# Roda cada tag dentro do seu próprio checkout limpo (tags_teste/<tag>), usando
# os scripts e os binários DA PRÓPRIA TAG.
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
hostname
nvidia-smi --query-gpu=name --format=csv,noheader

RAIZ=$PWD
for tag in v0.0-baseline v0.1-cpu-memoria v0.2-cpu-openmp v1.0-cuda-ingenuo \
           v1.1-cuda-residente v1.2-cuda-shared v1.3-relatorio; do
    cd "$RAIZ/tags_teste/$tag"
    echo "================ $tag ================"
    ultimo=legado
    for bin in cpu gpu; do
        [ -x ./$bin ] || continue
        ultimo=$bin
        echo "--- scripts/validar.sh ./legado ./$bin"
        scripts/validar.sh ./legado ./$bin | grep -E "^[<>] (Epoca|Acuracia|Perda)|VALIDACAO"
    done
    reps=3
    [ "$tag" = v1.0-cuda-ingenuo ] && reps=1
    echo "--- scripts/bench.sh ./$ultimo $reps"
    scripts/bench.sh ./$ultimo $reps | grep MEDIA
done
