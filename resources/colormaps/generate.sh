#!/bin/bash

while read l; do
	IFS=';' read -ra NAMECMAP <<< "$l"
	if [ ${#NAMECMAP[@]} -eq 2 ]; then
		name="${NAMECMAP[0]/ /_}"
		cmap="${NAMECMAP[1]}"
		gnuplot -e "set print '-'; set palette ${cmap}; show palette palette 256" | cut -b 60- > "cmaps/${name}"
		echo "cmaps/${name}"
	fi
done < definitions.txt