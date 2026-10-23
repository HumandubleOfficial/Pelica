#!/bin/sh
# Head-to-head on one prompt: Colibri's glm53 engine vs Pelica, each with its
# defaults, each in a fresh process, same token ids, greedy decode of N tokens.
# Usage: sh bench/vs_colibri.sh [N]
set -e
cd "$(dirname "$0")/.."
M=/c/models/glm53_flash_i4
COLI=${COLI:-"$LOCALAPPDATA/colibri/runtime/colibri-v2.0.0/glm53.exe"}
N=${1:-48}
P="Write a short paragraph about how hummingbirds hover."

# the prompt's ids from Pelica's tokenizer and chat template
./target/release/pelica.exe -m $M -p "$P" --no-think -n 0 --teacher > bench/ids.log 2>&1
IDS=$(tr '\r' '\n' < bench/ids.log | grep -a "prompt_ids" | sed 's/.*prompt_ids //' | tr ' ' ',')
echo "prompt ids: $IDS"

"$COLI" --model "$(cygpath -m $M)" --ids "$IDS" --greedy $N > bench/colibri.log 2>&1
./target/release/pelica.exe -m $M --ids "$IDS" -n $N --teacher > bench/pelica.log 2>&1

echo "== colibri"; tr '\r' '\n' < bench/colibri.log | grep -aE "^greedy|^decode|prefill [0-9]+ tokens"
echo "== pelica";  tr '\r' '\n' < bench/pelica.log  | grep -aE "^greedy|decode [0-9]+ tokens|prefill [0-9]+ tokens|decode experts"
