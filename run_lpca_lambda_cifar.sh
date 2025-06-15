#!/bin/bash

sizes=( 32 )

for lambda in $(seq 0.5 .5 2.5); do
    for k in ${sizes[@]}; do
        echo "$k,$lambda"
        python compute_encodings.py CIFAR 4 $k 1 False False $lambda 1000
    done
done
