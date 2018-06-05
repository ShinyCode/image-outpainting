#!/usr/bin/env bash

stdbuf -i0 -o0 -e0 python -u train_ld.py | tee output/out
