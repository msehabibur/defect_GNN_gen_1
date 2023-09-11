#!/bin/bash
B=`pwd`
for ((n=1; n<=312; n++))
do
    A=`head -$n $B/sys1 | tail -1 | cut -f 1`
    C=`head -$n $B/sys2 | tail -1 | cut -f 1`

    cd $B/$A/$C
    if [ ! -f OUTCAR ]; then
        echo "$B/$A/$C" >> $B/missing_folder
    else
        gzip -d OUTCAR.gz
        grep "free  energy   TOTEN  =" OUTCAR | tail -1 | awk '{print $5}' >> /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/energy_optimized
    fi
    cd $B
done
