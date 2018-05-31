#!/usr/bin/env bash

scp -r -i "cs_230_key.pem" ubuntu@34.237.120.117:~/image-outpainting/output/models $1
