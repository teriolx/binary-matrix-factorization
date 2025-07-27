#!/bin/bash

sizes=( 4 8 16 32 64 128 )

for lambda in $(seq 0.5 .5 2.5); do
    for k in ${sizes[@]}; do
        echo "$k,$lambda"
        python compute_encodings.py ZINC 4 $k 1 False False $lambda
    done
done
