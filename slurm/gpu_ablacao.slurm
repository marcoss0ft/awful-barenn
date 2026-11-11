#!/bin/bash
#SBATCH --job-name=aps2_ablacao
#SBATCH --output=resultados/gpu_ablacao_%j.out
#SBATCH --partition=sunny_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

# Uso: (no nó de login) module load cuda/12.8.1 && make gpu gpu_sem_padding gpu_pesos_global
#      sbatch slurm/gpu_ablacao.slurm
# Mede, intercalados no mesmo job, o kernel por época com pesos em memória
# compartilhada (gpu), o mesmo kernel sem a posição extra por linha
# (gpu_sem_padding) e com os pesos na memória global (gpu_pesos_global).
module load cuda/12.8.1
export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
hostname
nvidia-smi --query-gpu=name --format=csv,noheader

echo "== As tres variantes devem dar a mesma saida =="
./gpu | grep -v Tempo > resultados/saida_gpu_shared.txt
for b in gpu_sem_padding gpu_pesos_global; do
    ./$b | grep -v Tempo | diff resultados/saida_gpu_shared.txt - > /dev/null \
        && echo "$b: saida identica a ./gpu" || echo "$b: SAIDA DIFERENTE"
done

for r in $(seq 1 10); do
    for b in gpu gpu_sem_padding gpu_pesos_global; do
        echo "$b $(./$b | grep 'Tempo de treino' | awk '{print $4}')"
    done
done > resultados/gpu_ablacao_dados.txt
cat resultados/gpu_ablacao_dados.txt

echo "== Medias (10 rodadas intercaladas) =="
awk '{ n[$1]++; s[$1] += $2; q[$1] += $2 * $2 }
     END { for (k in n) { m = s[k] / n[k]
           printf "%-17s treino %.4f s (desvio %.4f)\n", k, m, sqrt((q[k] - n[k] * m * m) / (n[k] - 1)) } }' \
    resultados/gpu_ablacao_dados.txt | sort
