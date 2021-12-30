#!/bin/sh

file_name=memories.csv

rm $file_name
touch $file_name

# shellcheck disable=SC2039
for i in {1..8}
do
	mpirun -np "$i" --oversubscribe python3 shift_image.py "$1" $file_name
	echo "$i done"
done

cat $file_name
