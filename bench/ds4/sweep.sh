#!/bin/bash
# DeepSeek V4 A/B sweep: interleaved configs, prints prompt + decode speed and drive stats.
#
# usage: [N=tokens] [P=prompt | PF=prompt-file] [VGB=.. RGB=..] [COOL=secs] [EXE=path] bash sweep.sh name rounds "ENV1" "ENV2" ...
#   each "ENV" is a list of VAR=value settings for one config, e.g. "PELICA_LATE_K=5"
#   VGB / RGB pin the VRAM expert GB and RAM cache GB (free RAM/VRAM varies between runs)
#   CTX sets the context (the CLI default is 2048)
#   COOL pauses between runs (the drive slows after long back-to-back reads)
#
# Before each run the model's usage history (<model>.pelica-usage) is restored from
# usage.bak, so every run starts from the same expert placement. Make usage.bak first:
#   cp "$M.pelica-usage" usage.bak
# Logs go to <name>_<config>_<round>.out/.err next to this script (PELICA_PROF=1 output in .err).
# Never run two engines at once, and don't build while a sweep runs (CPU contention).
cd "$(dirname "$0")"
M="${MODEL:-C:/models/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF/DeepSeek-V4-Flash-Q4_K-0731.gguf}"
EXE="${EXE:-./pds4.exe}"
PROMPT="${P:-Write a detailed explanation of how a hash map works, then a complete example implementation in Rust with comments.}"
[ -n "$PF" ] && PROMPT="$(cat "$PF")"
name=$1; rounds=$2; shift 2
for r in $(seq 1 $rounds); do
  i=0
  for E in "$@"; do
    i=$((i+1))
    timeout ${COOL:-15} tail -f /dev/null; cp usage.bak "$M.pelica-usage"
    env PELICA_PROF=1 ${VGB:+PELICA_EXPERTS_GB=$VGB} ${RGB:+PELICA_RAM_GB=$RGB} $E "$EXE" -m "$M" --temp 0 ${CTX:+--ctx $CTX} -n ${N:-200} --no-think -p "$PROMPT" > ${name}_${i}_$r.out 2> ${name}_${i}_$r.err
    echo "[$E] r$r $(grep -o 'prefill [0-9]* tokens in [0-9.]*s ([0-9.]* tok/s)' ${name}_${i}_$r.err) | $(grep -o 'decode [0-9]* tokens.*tok/s' ${name}_${i}_$r.err) | $(grep -o 'read [0-9.]* GB.*' ${name}_${i}_$r.err | tail -1)"
  done
done
cp usage.bak "$M.pelica-usage"
