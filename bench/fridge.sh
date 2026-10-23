#!/bin/sh
# Decode benchmark on a prompt the routing history has not seen much.
# Usage: sh bench/fridge.sh <log name> [extra pelica args]
cd "$(dirname "$0")/.."
name=$1; shift
./target/release/pelica.exe -m /c/models/glm53_flash_i4 -p "Explain how a refrigerator works, step by step, for a curious teenager." --no-think -n 160 --cache-gb 19 --vram-gb 13 "$@" > bench/$name.log 2>&1
tr '\r' '\n' < bench/$name.log | grep -aE "tiers|decode|tier:"
