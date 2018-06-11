#!/usr/bin/env bash

rsync -av --ignore-existing -e "ssh -i cs_230_key.pem" ubuntu@34.237.120.117:~/image-outpainting/output/*.png $1
