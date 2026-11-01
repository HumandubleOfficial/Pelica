#!/bin/sh
# Teacher-forced check of the GPU-resident engine against llama.cpp: llama.cpp
# generates N greedy tokens (with its top-2 log-probs); Pelica reads the same
# sequence and reports its argmax at every position. Mismatches are listed
# with llama.cpp's own margin between its top two candidates.
# Usage: sh bench/oss_teacher.sh [N] [prompt]
cd "$(dirname "$0")/.."
M=${OSS_MODEL:-/c/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf}
N=${1:-128}
P=${2:-Write a short paragraph about how hummingbirds hover.}
LS=${LLAMA_SERVER:-$HOME/.unsloth/llama.cpp/build/bin/Release/llama-server.exe}
./target/release/pelica.exe -m "$M" -p "$P" -n 1 --teacher > bench/oss_t0.log 2>&1
IDS=$(grep -a "^prompt_ids" bench/oss_t0.log | sed 's/prompt_ids //' | tr ' ' ',')
("$LS" -m "$M" -c 4096 --port 8089 > /dev/null 2>&1 &)
for i in $(seq 1 120); do curl -s localhost:8089/health | grep -q ok && break; sleep 1; done
curl -s localhost:8089/completion -d "{\"prompt\":[${IDS}],\"n_predict\":$N,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false,\"ignore_eos\":true,\"n_probs\":2,\"return_tokens\":true}" > bench/oss_t_llama.json
taskkill //IM llama-server.exe //F > /dev/null 2>&1
ALL=$(python -c "import json;j=json.load(open('bench/oss_t_llama.json'));print(','.join(['$IDS']+[str(t) for t in j['tokens']]))")
./target/release/pelica.exe -m "$M" --ids "$ALL" -n 1 --teacher > bench/oss_t1.log 2>&1
python - "$IDS" <<'PY'
import json, sys
np_ = len(sys.argv[1].split(','))
j = json.load(open('bench/oss_t_llama.json'))
ref = j['tokens']
probs = j['completion_probabilities']
line = [l for l in open('bench/oss_t1.log', encoding='utf-8', errors='replace') if l.startswith('teacher')][0]
arg = [int(x) for x in line.split()[1:]]
greedy = [int(x) for x in [l for l in open('bench/oss_t1.log', encoding='utf-8', errors='replace') if l.startswith('greedy')][0].split()[1:]]
got = arg[np_ - 1:] + greedy[:1]   # argmax after the last prompt token predicts ref[0]
bad = [(i, got[i], ref[i]) for i in range(len(ref)) if got[i] != ref[i]]
print(f"teacher-forced: {len(ref) - len(bad)} of {len(ref)} positions agree")
for i, g, r in bad:
    top = probs[i]['top_logprobs']
    gap = top[0]['logprob'] - top[1]['logprob'] if len(top) > 1 else float('nan')
    alt = top[1]['id'] if len(top) > 1 else None
    print(f"  pos {i}: pelica {g}, llama.cpp {r} (llama.cpp top-2 margin {gap:.3f}, its 2nd choice {alt})")
PY
