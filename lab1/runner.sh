#!/bin/bash

BINARY="./.build/bin/sol_multy_comm"

printf "Processes\tTime(s)\n"

for NP in {1..6}; do
    START=$(date +%s.%N)
    mpirun -np $NP $BINARY
    END=$(date +%s.%N)

    ELAPSED=$(echo "$END - $START" | bc)
    printf "%d\t\t%.6f\n" $NP $ELAPSED
done