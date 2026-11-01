#!/bin/sh
# Interleaved comparison of several environment settings (one per argument),
# N rounds, on the head-to-head prompt with pinned tiers.
# Usage: sh bench/abc.sh N "ENV_A" "ENV_B" ...
cd "$(dirname "$0")/.."
n=$1; shift
for i in $(seq 1 $n); do
  for cfg in "$@"; do
    env $cfg bash bench/hb.sh ab_tmp > /dev/null
    r=$(tr '\r' '\n' < bench/ab_tmp.log | grep -aoE "decode [0-9]+ tokens in [0-9.]+s: [0-9.]+ tok/s" | grep -aoE "[0-9.]+ tok/s")
    d=$(tr '\r' '\n' < bench/ab_tmp.log | grep -aoE "disk [0-9.]+ GB/s, busy [0-9]+% at [0-9.]+ GB/s")
    t=$(diff -q <(grep -aE "^teacher|^greedy" bench/pelica_ref.log) <(tr '\r' '\n' < bench/ab_tmp.log | grep -aE "^teacher|^greedy") >/dev/null && echo same || echo DIFF)
    echo "$cfg | $r | $d | tokens $t"
  done
done
