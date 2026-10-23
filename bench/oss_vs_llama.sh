#!/bin/sh
# Pelica's GPU-resident engine vs llama.cpp (CUDA) on gpt-oss-20b: same
# prompt ids, greedy, N tokens; prints both speeds and whether the tokens match.
# Usage: sh bench/oss_vs_llama.sh [N] [prompt]
cd "$(dirname "$0")/.."
M=${OSS_MODEL:-/c/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf}
N=${1:-128}
P=${2:-Write a short paragraph about how hummingbirds hover.}
LS=${LLAMA_SERVER:-$HOME/.unsloth/llama.cpp/build/bin/Release/llama-server.exe}
./target/release/pelica.exe -m "$M" -p "$P" -n $N --teacher --ignore-eos > bench/oss_pelica.log 2>&1
IDS=$(grep -a "^prompt_ids" bench/oss_pelica.log | sed 's/prompt_ids //' | tr ' ' ',')
grep -a "decode" bench/oss_pelica.log
"$LS" -m "$M" -ngl 99 -c 4096 --port 8089 -fa on > bench/oss_llama_server.log 2>&1 &
for i in $(seq 1 120); do curl -s localhost:8089/health | grep -q ok && break; sleep 1; done
# warm-up, then the measured run
curl -s localhost:8089/completion -d "{\"prompt\":[${IDS}],\"n_predict\":8,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false}" > /dev/null
curl -s localhost:8089/completion -d "{\"prompt\":[${IDS}],\"n_predict\":$N,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false,\"ignore_eos\":true,\"return_tokens\":true}" > bench/oss_llama.json
kill %1 2>/dev/null; sleep 1
python - <<'PY'
import json
j = json.load(open('bench/oss_llama.json'))
t = j['timings']
print(f"[llama.cpp] prefill {t['prompt_n']} tokens: {t['prompt_per_second']:.1f} tok/s; decode {t['predicted_n']} tokens: {t['predicted_per_second']:.1f} tok/s")
ref = j['tokens']
got = [int(x) for x in [l for l in open('bench/oss_pelica.log', encoding='utf-8', errors='replace') if l.startswith('greedy')][0].split()[1:]]
n = min(len(ref), len(got))
same = next((i for i in range(n) if ref[i] != got[i]), n)
print(f"tokens: first {same} of {n} identical" + ("" if same == n else f" (pelica {got[same]} vs llama.cpp {ref[same]} at {same})"))
PY
