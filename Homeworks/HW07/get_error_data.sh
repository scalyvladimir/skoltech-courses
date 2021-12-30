#!/bin/sh

file_name=errors.csv

rm $file_name
touch $file_name

# shellcheck disable=SC2039
for i in {1..8}
do
	# shellcheck disable=SC2004
	mpirun -np 8 --oversubscribe python3 int.py $((10**$i)) $file_name
done

cat $file_name
