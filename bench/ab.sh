#!/bin/sh
# Interleaved A/B on the head-to-head prompt with pinned tiers: alternates two
# environment settings N times and prints tok/s and disk stats per run.
# Usage: sh bench/ab.sh N "ENV_A" "ENV_B"   (e.g. "PELICA_AIO=0" "PELICA_AIO=1")
cd "$(dirname "$0")/.."
n=$1; a=$2; b=$3
for i in $(seq 1 $n); do
  for cfg in "$a" "$b"; do
    env $cfg bash bench/hb.sh ab_tmp > /dev/null
    r=$(tr '\r' '\n' < bench/ab_tmp.log | grep -aoE "decode [0-9]+ tokens in [0-9.]+s: [0-9.]+ tok/s" | grep -aoE "[0-9.]+ tok/s")
    d=$(tr '\r' '\n' < bench/ab_tmp.log | grep -aoE "disk [0-9.]+ GB/s, busy [0-9]+% at [0-9.]+ GB/s")
    t=$(diff -q <(grep -aE "^teacher|^greedy" bench/pelica_ref.log) <(tr '\r' '\n' < bench/ab_tmp.log | grep -aE "^teacher|^greedy") >/dev/null && echo same || echo DIFF)
    echo "$cfg | $r | $d | tokens $t"
  done
done
