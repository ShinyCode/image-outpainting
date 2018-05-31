#!/usr/bin/env bash

scp -i "cs_230_key.pem" ubuntu@34.237.120.117:~/image-outpainting/output/*.png $1
