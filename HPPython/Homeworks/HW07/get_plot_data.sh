#!/bin/sh

file_name=times.csv

rm $file_name
touch $file_name

# shellcheck disable=SC2039
for i in {1..8}
do
	mpirun -np "$i" --oversubscribe python3 int.py $((10**8)) $file_name
done

cat $file_name
