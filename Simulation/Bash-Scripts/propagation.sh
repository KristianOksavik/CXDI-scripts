#!/bin/sh

for i in $(seq 1 200);#$(seq 1 200);
do
    num=$(($i))
    python ePIE-Reconstruction.py $num
done
#python moviemaker.py "/cluster/home/kristaok/reconstruction/tempimages/"
