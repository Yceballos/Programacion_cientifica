#!/bin/bash
#
Script="/home/programacion3/yceballos/Proyecto/processing.py"
cores=${ARG1}
library="/home/programacion3/yceballos/env/lib/python3.7/site-packages"

for n in {1..4}
do
        arg="/home/programacion3/yceballos/Proyecto/data/lexicon${n}.txt"
        Name="Completado_lexicon${n}"
	echo "$Script '$arg' 8" -s $library | qsub -N $Name -l nodes=1:ppn=5

done



