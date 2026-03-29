#!/bin/bash
cd ./.build/bin/
EXECUTABLES=("sol_multy_comm" "sol_multy_split")
MAX_CORES=6

for exe in "${EXECUTABLES[@]}"; do
    filename=$(basename "$exe")
    output_file="${filename}-res.csv"
    
    echo "Cores,Execution_Time_Sec" > "$output_file"
    
    for ((x=1; x<=MAX_CORES; x++)); do
        start_time=$(date +%s.%N)
        
        mpirun -np "$x" "$exe"
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        
        echo "${x},${duration}" >> "$output_file"
    done
done