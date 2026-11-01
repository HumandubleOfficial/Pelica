#!/bin/sh
# Pelica alone on the head-to-head prompt (ids from bench/ids.log), checks the
# tokens against bench/pelica_ref.log. Usage: sh bench/hb.sh <log name> [args]
cd "$(dirname "$0")/.."
name=$1; shift
IDS=$(tr '\r' '\n' < bench/ids.log | grep -a "prompt_ids" | sed 's/.*prompt_ids //' | tr ' ' ',')
./target/release/pelica.exe -m /c/models/glm53_flash_i4 --ids "$IDS" -n 48 --teacher --vram-gb 13.4 --cache-gb 23 "$@" > bench/$name.log 2>&1
tr '\r' '\n' < bench/$name.log | grep -aE "tiers|decode [0-9]|decode time|waits|profile"
if diff -q <(grep -aE "^teacher|^greedy" bench/pelica_ref.log) <(tr '\r' '\n' < bench/$name.log | grep -aE "^teacher|^greedy") > /dev/null; then echo "tokens: identical"; else echo "tokens: DIFFERENT"; fi
