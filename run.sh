#!/bin/bash
cd "$(dirname "$0")"
name=$1; cipher=$2; gs=$3; ek=$4; thr=$5; n=$6; shift 6
out=out/$name; mkdir -p $out
${ANNEAL:-./anneal} $cipher $out R=128 thr=$thr "$@" > $out/anneal.log 2>&1 &
pid=$!
s2() { 
  for d in $out/order-*.done; do
    [ -e "$d" ] || continue
    b=${d%.done}
    [ -e $b.s2top ] && continue
    ./wordcheck ${VOCAB:-data/vocab_all.txt} data/keys-strict.txt $gs $ek $n < $b.tsv > $b.s2.tmp
    { awk -F'\t' '$1>=12 || ($3==1 && $1>=8) || $2>=6' $b.s2.tmp; head -200 $b.s2.tmp; } | sort -u > $b.s2top
    rm -f $b.s2.tmp
  done
}
while kill -0 $pid 2>/dev/null; do s2; sleep 15; done
wait $pid; echo "anneal exit $?" >> $out/anneal.log
s2
echo done > $out/ALL_DONE
