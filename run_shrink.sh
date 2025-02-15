#!/bin/bash

FILEPATH="/home/bolzo/puffinn-tests/clann/datasets/glove-25-angular.hdf5"

for DIM in $(seq 3 2 25); do
    echo "Running shrink.py with dim=$DIM"
    python3 shrink.py -f "$FILEPATH" -dim "$DIM"
done
