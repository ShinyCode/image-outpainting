#!/usr/bin/env bash

stdbuf -i0 -o0 -e0 python -u train.py | tee output/out
