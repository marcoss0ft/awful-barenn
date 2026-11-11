#!/bin/bash
# Uso (no nó de login do Franky, dentro de um clone do repositório):
#     module load cuda/12.8.1 && scripts/compilar_tags.sh
#
# Para cada tag, faz checkout em uma pasta temporária (git worktree), compila
# o binário principal daquela versão com o Makefile DA PRÓPRIA TAG e guarda o
# resultado em bin_tags/<tag>. Mostra que cada marco compila de forma
# independente e permite medir todas as versões no mesmo job.
set -e
mkdir -p bin_tags

compilar() {
    local tag=$1 alvo=$2
    local pasta
    pasta=$(mktemp -d)
    git worktree add --detach "$pasta" "$tag" > /dev/null
    make -C "$pasta" "$alvo" > /dev/null
    cp "$pasta/$alvo" "bin_tags/$tag"
    git worktree remove --force "$pasta"
    echo "$tag: make $alvo -> bin_tags/$tag"
}

compilar v0.0-baseline       legado
compilar v0.1-cpu-memoria    cpu
compilar v0.2-cpu-openmp     cpu
compilar v1.0-cuda-ingenuo   gpu
compilar v1.1-cuda-residente gpu
compilar v1.2-cuda-shared    gpu
