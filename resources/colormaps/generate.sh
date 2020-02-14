#!/bin/bash

while read -r l || [ -n "$l" ]; do
	IFS=';' read -ra NAMECMAP <<< "$l"
	if [ ${#NAMECMAP[@]} -eq 2 ]; then
		name="${NAMECMAP[0]/ /_}"
		cmap="${NAMECMAP[1]}"
		echo "cmaps/${name} :"
		gnuplot -e "set print '-'; set palette ${cmap}; show palette palette 256" | cut -b 60- > "cmaps/${name}"
	fi
done < definitions.txt

git add cmaps/*
