#!/bin/sh
# Like hb.sh but with default tier sizes (what is free at start), so the VRAM
# growth after prefill and the RAM margin take part. Usage: sh bench/def.sh <log name> [args]
cd "$(dirname "$0")/.."
name=$1; shift
IDS=$(tr '\r' '\n' < bench/ids.log | grep -a "prompt_ids" | sed 's/.*prompt_ids //' | tr ' ' ',')
./target/release/pelica.exe -m /c/models/glm53_flash_i4 --ids "$IDS" -n 48 --teacher "$@" > bench/$name.log 2>&1
tr '\r' '\n' < bench/$name.log | grep -aE "tiers|more VRAM|decode [0-9]|disk [0-9]"
if diff -q <(grep -aE "^teacher|^greedy" bench/pelica_ref.log) <(tr '\r' '\n' < bench/$name.log | grep -aE "^teacher|^greedy") > /dev/null; then echo "tokens: identical"; else echo "tokens: DIFFERENT"; fi
