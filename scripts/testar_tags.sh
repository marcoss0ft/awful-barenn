#!/bin/bash
# Uso (no nó de login do Franky, dentro de um clone do repositório):
#     module load cuda/12.8.1 && scripts/testar_tags.sh
#     sbatch slurm/testar_tags.slurm
#
# Prepara cada tag do jeito que alguém que acabou de clonar faria: checkout
# limpo em tags_teste/<tag>, chmod +x nos scripts (até a v1.3 eles estão sem o
# bit de execução no git) e make dos alvos que aquela tag tem.
set -e
for pasta in tags_teste/*/; do
    [ -d "$pasta" ] && git worktree remove --force "$pasta"
done
mkdir -p tags_teste

preparar() {
    local tag=$1
    shift
    git worktree add --detach "tags_teste/$tag" "$tag" > /dev/null
    chmod +x "tags_teste/$tag"/scripts/*.sh
    make -C "tags_teste/$tag" "$@" > /dev/null
    echo "$tag: chmod +x scripts/*.sh && make $*"
}

preparar v0.0-baseline       legado
preparar v0.1-cpu-memoria    legado cpu
preparar v0.2-cpu-openmp     legado cpu
preparar v1.0-cuda-ingenuo   legado cpu gpu
preparar v1.1-cuda-residente legado cpu gpu
preparar v1.2-cuda-shared    legado cpu gpu
preparar v1.3-relatorio      legado cpu gpu
