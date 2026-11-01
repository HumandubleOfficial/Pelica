#!/bin/sh
M=/c/models/glm53_flash_i4
P="Write a short paragraph about how hummingbirds hover."
run() { name=$1; shift; ./target/release/pelica.exe -m $M -p "$P" --no-think -n 48 --teacher "$@" > bench/$name.log 2>&1; }
run direct  --no-warm --direct

run warm    --direct
