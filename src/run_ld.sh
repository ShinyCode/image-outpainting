#!/usr/bin/env bash

# Runs train_ld.py and saves the console output to output/out
stdbuf -i0 -o0 -e0 python -u train_ld.py | tee output/out
