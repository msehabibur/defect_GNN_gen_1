#!/bin/bash
# when the XDATCAR file has just atoms moving, not the cell volume (i.e. ISIF=2 condition)
B=`pwd`
for ((  n = 1;  n <= 312;  n++  ))
do
        A=`head -$n $B/sys1 | tail -1 | cut -f 1`
        C=`head -$n $B/sys2 | tail -1 | cut -f 1`
        D=`head -$n $B/sysNum | tail -1 | cut -f 1`

        cd $B/$A/$C/

        # script to extract intermediate structures
        head -7 XDATCAR > top
        sed '1,7d' XDATCAR > topRemoved

        totalAtoms=`sed -n 7p XDATCAR | tr ' ' '\n' | awk '{ sum+=$1} END {print sum}'`
        totalLines=`cat topRemoved | wc -l`
        totalStr=`awk "BEGIN {print $totalLines/($totalAtoms+1)}"`
        linesPerStr=`awk "BEGIN {print $totalLines/$totalStr}"`

        # saves intermediate structures
        mkdir intStructs
        for ((i=0; i<$totalStr;i++))
                do
                begin_line=$(($(($linesPerStr * $i))+1))
                end_line=$(($linesPerStr * $(($i+1))))

                sed -n $begin_line,$(($end_line))p topRemoved > POSCAR_temp
                cat top POSCAR_temp >intStructs/$D-$A-$C-POSCAR$i-Neutral.vasp
                rm POSCAR_temp

        # Copying all intermediate structures to a directory
        cp $B/$A/$C/ intStructs/* /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/int_structures
        
        # Gathering the file names
        echo $D-$A-$C-POSCAR$i-Neutral.vasp >> /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/intStructsNames
        #echo $B/$A/$C/ >> /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/intStructsDirectory
        done
        rm top
        rm topRemoved
    echo $totalStr >> /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/int_struct_freq
    grep "free  energy   TOTEN  =" OUTCAR | tail -n $totalStr | awk '{print $5}' >> /depot/amannodi/data/Habibur_calcs/34_ZB_Semiconductors_GNN/Native/Neutral/intStructsToten
    rm -r intStructs

done

