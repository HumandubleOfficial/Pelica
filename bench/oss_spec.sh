#!/bin/sh
# Speculative decoding check: each prompt with drafting off and on; the
# text must be identical, and the speed and tokens per pass are reported.
# Usage: sh bench/oss_spec.sh [N]
cd "$(dirname "$0")/.."
M=${OSS_MODEL:-/c/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf}
N=${1:-400}
i=0
while IFS= read -r P; do
  i=$((i+1))
  for d in 0 7; do
    PELICA_DRAFT=$d PELICA_STATS=1 ./target/release/pelica.exe -m "$M" -p "$P" -n "$N" > bench/spec_${i}_$d.out 2> bench/spec_${i}_$d.err
  done
  same=$(cmp -s bench/spec_${i}_0.out bench/spec_${i}_7.out && echo same || echo DIFFERENT)
  a=$(grep -a "decode" bench/spec_${i}_0.err | sed 's/.*: \([0-9.]*\) tok\/s.*/\1/')
  b=$(grep -a "decode" bench/spec_${i}_7.err | sed 's/.*: \([0-9.]*\) tok\/s (\([0-9]*\) passes, \([0-9.]*\) tokens each).*/\1 tok\/s, \3 per pass/')
  n=$(grep -a "decode" bench/spec_${i}_7.err | sed 's/.*decode \([0-9]*\) tokens.*/\1/')
  echo "prompt $i ($n tokens): no draft $a tok/s | draft $b | text $same"
done <<'PROMPTS'
Write a 300-word essay about the history of the bicycle.
Write a Python function that parses a CSV file into a list of dictionaries, with error handling and docstrings.
Here is a list: apple banana cherry date elderberry fig grape. Repeat the list exactly three times, separated by newlines.
Rewrite this paragraph, changing only the word "quick" to "slow": The quick brown fox jumps over the lazy dog. The dog, startled, barks loudly and runs into the forest, where it hides behind a large oak tree until the fox has gone away.
PROMPTS
