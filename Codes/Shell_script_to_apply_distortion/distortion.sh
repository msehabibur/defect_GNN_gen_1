#!/bin/bash

input_file="POSCAR.cif"
output_prefix="disturbed"

disturbance_values=($(seq 0 0.0005 0.3))  # Generate a sequence from 0 to 1 with a step size of 0.005

for disturbance in "${disturbance_values[@]}"; do
    output_file="${output_prefix}_${disturbance}.cif"
    atomsk $input_file -disturb $disturbance $output_file
done
