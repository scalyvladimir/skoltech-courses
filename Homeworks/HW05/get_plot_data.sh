#!/bin/sh

rm results.csv
touch results.csv

for i in {1..8}
do
  mpirun -n $i python3 bif.py   
done
cat results.csv
