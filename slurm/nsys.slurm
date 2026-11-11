#!/bin/bash
#SBATCH --job-name=aps2_nsys
#SBATCH --output=resultados/nsys_%j.out
#SBATCH --partition=sunny_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:15:00

# Uso: sbatch slurm/nsys.slurm <nome_do_relatorio> [segundos]
# Linha do tempo CUDA com o Nsight Systems (aula profile-nvidia). Com [segundos]
# a captura para depois desse tempo (a versão ingênua geraria um relatório enorme).
# O resumo em texto (nsys stats) vai para o .out; o .nsys-rep abre no nsys-ui.
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
NOME=${1:-perfil_gpu}
DURACAO=${2:-}

OPCAO=""
if [ -n "$DURACAO" ]; then OPCAO="--duration=$DURACAO"; fi
nsys profile --trace=cuda --sample=none $OPCAO --force-overwrite=true \
    --output=resultados/$NOME ./gpu

# Tempo das chamadas da API CUDA, dos kernels e das cópias (tempo e volume)
nsys stats --force-export=true \
    --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
    resultados/$NOME.nsys-rep
