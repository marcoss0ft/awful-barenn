#!/bin/bash
# Uso: scripts/validar.sh ./binario_referencia ./binario_novo
# Compara a saída dos dois programas ignorando as linhas de tempo.
# Saída idêntica => perdas por época, acurácia, inferências e confusões iguais.
REF=$1
NOVO=$2

"$REF"  | grep -v "Tempo" > /tmp/ref_$$.txt
"$NOVO" | grep -v "Tempo" > /tmp/novo_$$.txt

if diff /tmp/ref_$$.txt /tmp/novo_$$.txt; then
    echo "VALIDACAO OK: saida identica a $REF (exceto linhas de tempo)"
    status=0
else
    echo "VALIDACAO: saidas diferentes (diff acima)"
    status=1
fi
rm -f /tmp/ref_$$.txt /tmp/novo_$$.txt
exit $status
